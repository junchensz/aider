import colorsys
import math
import os
import random
import shutil
import sqlite3
import sys
import time
import warnings
from collections import Counter, defaultdict, namedtuple
from importlib import resources
from pathlib import Path

from diskcache import Cache
from grep_ast import TreeContext, filename_to_lang
from pygments.lexers import guess_lexer_for_filename
from pygments.token import Token
from tqdm import tqdm

from aider.dump import dump
from aider.special import filter_important_files
from aider.utils import Spinner

# tree_sitter is throwing a FutureWarning
warnings.simplefilter("ignore", category=FutureWarning)
from tree_sitter_languages import get_language, get_parser  # noqa: E402

Tag = namedtuple("Tag", "rel_fname fname line name kind".split())


SQLITE_ERRORS = (sqlite3.OperationalError, sqlite3.DatabaseError, OSError)


class RepoMap:
    CACHE_VERSION = 3
    TAGS_CACHE_DIR = f".aider.tags.cache.v{CACHE_VERSION}"

    warned_files = set()

    def __init__(
        self,
        map_tokens=1024,
        root=None,
        main_model=None,
        io=None,
        repo_content_prefix=None,
        verbose=False,
        max_context_window=None,
        map_mul_no_files=8,
        refresh="auto",
    ):
        self.io = io
        self.verbose = verbose
        self.refresh = refresh

        if not root:
            root = os.getcwd()
        self.root = root

        self.load_tags_cache()
        self.cache_threshold = 0.95

        self.max_map_tokens = map_tokens
        self.map_mul_no_files = map_mul_no_files
        self.max_context_window = max_context_window

        self.repo_content_prefix = repo_content_prefix

        self.main_model = main_model

        self.tree_cache = {}
        self.tree_context_cache = {}
        self.map_cache = {}
        self.map_processing_time = 0
        self.last_map = None

        if self.verbose:
            self.io.tool_output(
                f"RepoMap initialized with map_mul_no_files: {self.map_mul_no_files}"
            )

    def token_count(self, text):
        len_text = len(text)
        if len_text < 200:
            return self.main_model.token_count(text)

        lines = text.splitlines(keepends=True)
        num_lines = len(lines)
        step = num_lines // 100 or 1
        lines = lines[::step]
        sample_text = "".join(lines)
        sample_tokens = self.main_model.token_count(sample_text)
        est_tokens = sample_tokens / len(sample_text) * len_text
        return est_tokens

    def get_repo_map(
        self,
        chat_files,
        other_files,
        mentioned_fnames=None,
        mentioned_idents=None,
        force_refresh=False,
    ):
        # 如果最大映射令牌数小于等于0，则直接返回
        if self.max_map_tokens <= 0:
            return
        # 如果没有其他文件，则直接返回
        if not other_files:
            return
        # 如果没有提到的文件名，则初始化为空集合
        if not mentioned_fnames:
            mentioned_fnames = set()
        # 如果没有提到的标识符，则初始化为空集合
        if not mentioned_idents:
            mentioned_idents = set()

        max_map_tokens = self.max_map_tokens

        # 如果没有聊天文件，则提供整个仓库的更大视图
        padding = 4096
        if max_map_tokens and self.max_context_window:
            target = min(
                int(max_map_tokens * self.map_mul_no_files),
                self.max_context_window - padding,
            )
        else:
            target = 0
        # 如果没有聊天文件且目标大于0，则更新最大映射令牌数
        if not chat_files and self.max_context_window and target > 0:
            max_map_tokens = target

        try:
            # 获取排名的标签映射
            files_listing = self.get_ranked_tags_map(
                chat_files,
                other_files,
                max_map_tokens,
                mentioned_fnames,
                mentioned_idents,
                force_refresh,
            )
        except RecursionError:
            # 如果发生递归错误，禁用仓库映射并输出错误信息
            self.io.tool_error("Disabling repo map, git repo too large?")
            self.max_map_tokens = 0
            return

        # 如果文件列表为空，则返回
        if not files_listing:
            return

        # 如果是详细模式，输出令牌数量
        if self.verbose:
            num_tokens = self.token_count(files_listing)
            self.io.tool_output(f"Repo-map: {num_tokens / 1024:.1f} k-tokens")

        # 根据是否有聊天文件，设置其他文件的前缀
        if chat_files:
            other = "other "
        else:
            other = ""

        # 如果有仓库内容前缀，则格式化并添加到仓库内容中
        if self.repo_content_prefix:
            repo_content = self.repo_content_prefix.format(other=other)
        else:
            repo_content = ""

        # 将文件列表添加到仓库内容中
        repo_content += files_listing

        # 返回示例: repo_content 可能是一个包含文件路径和相关信息的字符串
        # 例如: "src/main.py\nsrc/utils.py\n"
        return repo_content
    def get_rel_fname(self, fname):
        try:
            return os.path.relpath(fname, self.root)
        except ValueError:
            # Issue #1288: ValueError: path is on mount 'C:', start on mount 'D:'
            # Just return the full fname.
            return fname

    def tags_cache_error(self, original_error=None):
        """Handle SQLite errors by trying to recreate cache, falling back to dict if needed"""

        if self.verbose and original_error:
            self.io.tool_warning(f"Tags cache error: {str(original_error)}")

        if isinstance(getattr(self, "TAGS_CACHE", None), dict):
            return

        path = Path(self.root) / self.TAGS_CACHE_DIR

        # Try to recreate the cache
        try:
            # Delete existing cache dir
            if path.exists():
                shutil.rmtree(path)

            # Try to create new cache
            new_cache = Cache(path)

            # Test that it works
            test_key = "test"
            new_cache[test_key] = "test"
            _ = new_cache[test_key]
            del new_cache[test_key]

            # If we got here, the new cache works
            self.TAGS_CACHE = new_cache
            return

        except SQLITE_ERRORS as e:
            # If anything goes wrong, warn and fall back to dict
            self.io.tool_warning(
                f"Unable to use tags cache at {path}, falling back to memory cache"
            )
            if self.verbose:
                self.io.tool_warning(f"Cache recreation error: {str(e)}")

        self.TAGS_CACHE = dict()

    def load_tags_cache(self):
        path = Path(self.root) / self.TAGS_CACHE_DIR
        try:
            self.TAGS_CACHE = Cache(path)
        except SQLITE_ERRORS as e:
            self.tags_cache_error(e)

    def save_tags_cache(self):
        pass

    def get_mtime(self, fname):
        try:
            return os.path.getmtime(fname)
        except FileNotFoundError:
            self.io.tool_warning(f"File not found error: {fname}")

    def get_tags(self, fname, rel_fname):
        """
        获取文件的标签信息,支持缓存机制
        
        Args:
            fname: 文件的完整路径
            rel_fname: 相对于仓库根目录的路径
            
        Returns:
            list: 包含文件中定义和引用的标签列表
        """
        # 获取文件的修改时间
        file_mtime = self.get_mtime(fname)
        if file_mtime is None:
            return []

        # 使用文件路径作为缓存键
        cache_key = fname
        try:
            # 尝试从缓存中获取数据
            val = self.TAGS_CACHE.get(cache_key)  # Issue #1308
        except SQLITE_ERRORS as e:
            # 如果发生SQLite错误,重置缓存并重试
            self.tags_cache_error(e)
            val = self.TAGS_CACHE.get(cache_key)

        # 如果缓存命中且文件未修改,直接返回缓存数据
        if val is not None and val.get("mtime") == file_mtime:
            try:
                return self.TAGS_CACHE[cache_key]["data"]
            except SQLITE_ERRORS as e:
                # 处理SQLite错误并重试
                self.tags_cache_error(e)
                return self.TAGS_CACHE[cache_key]["data"]

        # 缓存未命中,解析文件获取标签
        data = list(self.get_tags_raw(fname, rel_fname))

        # 更新缓存
        try:
            self.TAGS_CACHE[cache_key] = {"mtime": file_mtime, "data": data}
            self.save_tags_cache()
        except SQLITE_ERRORS as e:
            # 处理SQLite错误,使用内存缓存
            self.tags_cache_error(e)
            self.TAGS_CACHE[cache_key] = {"mtime": file_mtime, "data": data}

        return data

    def get_tags_raw(self, fname, rel_fname):
        lang = filename_to_lang(fname)
        if not lang:
            return

        try:
            language = get_language(lang)
            parser = get_parser(lang)
        except Exception as err:
            print(f"Skipping file {fname}: {err}")
            return

        query_scm = get_scm_fname(lang)
        if not query_scm.exists():
            return
        query_scm = query_scm.read_text()

        code = self.io.read_text(fname)
        if not code:
            return
        tree = parser.parse(bytes(code, "utf-8"))

        # Run the tags queries
        query = language.query(query_scm)
        captures = query.captures(tree.root_node)

        captures = list(captures)

        saw = set()
        for node, tag in captures:
            if tag.startswith("name.definition."):
                kind = "def"
            elif tag.startswith("name.reference."):
                kind = "ref"
            else:
                continue

            saw.add(kind)

            result = Tag(
                rel_fname=rel_fname,
                fname=fname,
                name=node.text.decode("utf-8"),
                kind=kind,
                line=node.start_point[0],
            )

            yield result

        if "ref" in saw:
            return
        if "def" not in saw:
            return

        # We saw defs, without any refs
        # Some tags files only provide defs (cpp, for example)
        # Use pygments to backfill refs

        try:
            lexer = guess_lexer_for_filename(fname, code)
        except Exception:  # On Windows, bad ref to time.clock which is deprecated?
            # self.io.tool_error(f"Error lexing {fname}")
            return

        tokens = list(lexer.get_tokens(code))
        tokens = [token[1] for token in tokens if token[0] in Token.Name]

        for token in tokens:
            yield Tag(
                rel_fname=rel_fname,
                fname=fname,
                name=token,
                kind="ref",
                line=-1,
            )

    def get_ranked_tags(
        self, chat_fnames, other_fnames, mentioned_fnames, mentioned_idents, progress=None
    ):
        """
        获取并排序代码标签。该方法使用PageRank算法分析代码库中的标识符定义和引用关系,生成一个按重要性排序的标签列表。

        参数:
        - chat_fnames: 当前聊天中涉及的文件名列表,这些文件会获得更高的权重
        - other_fnames: 其他需要分析的文件名列表
        - mentioned_fnames: 在对话中提到的文件名列表,这些文件会获得更高的权重
        - mentioned_idents: 在对话中提到的标识符列表,这些标识符的定义和引用会获得更高的权重
        - progress: 可选的进度回调函数,用于显示处理进度

        返回:
        - ranked_tags: 按重要性排序的标签列表,每个标签包含文件名、标识符名称、类型和行号等信息

        工作流程:
        1. 收集所有文件中的标识符定义和引用
        2. 构建有向图,节点是文件,边表示标识符的定义-引用关系
        3. 使用PageRank算法计算每个文件的重要性
        4. 根据文件重要性和标识符使用情况对标签进行排序
        5. 返回排序后的标签列表
        """
        import networkx as nx

        # 初始化数据结构来存储定义和引用信息
        defines = defaultdict(set)  # 存储标识符定义所在的文件
        references = defaultdict(list)  # 存储标识符被引用的文件
        definitions = defaultdict(set)  # 存储具体的定义标签
        
        # 初始化PageRank的个性化向量
        personalization = dict()

        # 合并所有需要分析的文件名
        fnames = set(chat_fnames).union(set(other_fnames))
        chat_rel_fnames = set()  # 存储聊天相关的相对文件名
        fnames = sorted(fnames)

        # 计算默认的个性化权重值
        personalize = 100 / len(fnames)

        # 获取标签缓存大小
        try:
            cache_size = len(self.TAGS_CACHE)
        except SQLITE_ERRORS as e:
            self.tags_cache_error(e)
            cache_size = len(self.TAGS_CACHE)

        # 对于大型仓库,显示进度条
        if len(fnames) - cache_size > 100:
            self.io.tool_output(
                "Initial repo scan can be slow in larger repos, but only happens once."
            )
            fnames = tqdm(fnames, desc="Scanning repo")
            showing_bar = True
        else:
            showing_bar = False

        # 遍历每个文件,收集标签信息
        for fname in fnames:
            if self.verbose:
                self.io.tool_output(f"Processing {fname}")
            if progress and not showing_bar:
                progress()

            # 检查文件是否存在
            try:
                file_ok = Path(fname).is_file()
            except OSError:
                file_ok = False

            if not file_ok:
                if fname not in self.warned_files:
                    self.io.tool_warning(f"Repo-map can't include {fname}")
                    self.io.tool_output(
                        "Has it been deleted from the file system but not from git?"
                    )
                    self.warned_files.add(fname)
                continue

            # 获取相对文件路径
            rel_fname = self.get_rel_fname(fname)

            # 设置个性化权重
            if fname in chat_fnames:
                personalization[rel_fname] = personalize
                chat_rel_fnames.add(rel_fname)

            if rel_fname in mentioned_fnames:
                personalization[rel_fname] = personalize

            # 获取并处理文件的标签
            tags = list(self.get_tags(fname, rel_fname))
            if tags is None:
                continue

            # 收集定义和引用信息
            for tag in tags:
                if tag.kind == "def":
                    defines[tag.name].add(rel_fname)
                    key = (rel_fname, tag.name)
                    definitions[key].add(tag)
                elif tag.kind == "ref":
                    references[tag.name].append(rel_fname)

        # 如果没有引用信息,使用定义信息作为引用
        if not references:
            references = dict((k, list(v)) for k, v in defines.items())

        # 获取同时存在定义和引用的标识符
        idents = set(defines.keys()).intersection(set(references.keys()))

        # 构建有向图
        G = nx.MultiDiGraph()

        # 添加边和权重
        for ident in idents:
            if progress:
                progress()

            definers = defines[ident]
            # 根据标识符特征设置权重乘数
            if ident in mentioned_idents:
                mul = 10
            elif ident.startswith("_"):
                mul = 0.1
            else:
                mul = 1

            # 为每个引用-定义对添加边
            for referencer, num_refs in Counter(references[ident]).items():
                for definer in definers:
                    num_refs = math.sqrt(num_refs)  # 缩放引用次数
                    G.add_edge(referencer, definer, weight=mul * num_refs, ident=ident)

        # 配置PageRank参数
        if personalization:
            pers_args = dict(personalization=personalization, dangling=personalization)
        else:
            pers_args = dict()

        # 计算PageRank值
        try:
            ranked = nx.pagerank(G, weight="weight", **pers_args)
        except ZeroDivisionError:
            try:
                ranked = nx.pagerank(G, weight="weight")
            except ZeroDivisionError:
                return []

        # 计算每个定义的排名
        ranked_definitions = defaultdict(float)
        for src in G.nodes:
            if progress:
                progress()

            src_rank = ranked[src]
            total_weight = sum(data["weight"] for _src, _dst, data in G.out_edges(src, data=True))
            
            # 分配排名值到每个出边
            for _src, dst, data in G.out_edges(src, data=True):
                data["rank"] = src_rank * data["weight"] / total_weight
                ident = data["ident"]
                ranked_definitions[(dst, ident)] += data["rank"]

        # 生成排序后的标签列表
        ranked_tags = []
        ranked_definitions = sorted(
            ranked_definitions.items(), reverse=True, key=lambda x: (x[1], x[0])
        )

        # 添加标签到结果列表
        for (fname, ident), rank in ranked_definitions:
            if fname in chat_rel_fnames:
                continue
            ranked_tags += list(definitions.get((fname, ident), []))

        # 处理没有标签的文件
        rel_other_fnames_without_tags = set(self.get_rel_fname(fname) for fname in other_fnames)
        fnames_already_included = set(rt[0] for rt in ranked_tags)

        # 根据PageRank值添加文件
        top_rank = sorted([(rank, node) for (node, rank) in ranked.items()], reverse=True)
        for rank, fname in top_rank:
            if fname in rel_other_fnames_without_tags:
                rel_other_fnames_without_tags.remove(fname)
            if fname not in fnames_already_included:
                ranked_tags.append((fname,))

        # 添加剩余的文件
        for fname in rel_other_fnames_without_tags:
            ranked_tags.append((fname,))

        return ranked_tags

    def get_ranked_tags_map(
        self,
        chat_fnames,
        other_fnames=None,
        max_map_tokens=None,
        mentioned_fnames=None,
        mentioned_idents=None,
        force_refresh=False,
    ):
        # Create a cache key
        cache_key = [
            tuple(sorted(chat_fnames)) if chat_fnames else None,
            tuple(sorted(other_fnames)) if other_fnames else None,
            max_map_tokens,
        ]

        if self.refresh == "auto":
            cache_key += [
                tuple(sorted(mentioned_fnames)) if mentioned_fnames else None,
                tuple(sorted(mentioned_idents)) if mentioned_idents else None,
            ]
        cache_key = tuple(cache_key)

        use_cache = False
        if not force_refresh:
            if self.refresh == "manual" and self.last_map:
                return self.last_map

            if self.refresh == "always":
                use_cache = False
            elif self.refresh == "files":
                use_cache = True
            elif self.refresh == "auto":
                use_cache = self.map_processing_time > 1.0

            # Check if the result is in the cache
            if use_cache and cache_key in self.map_cache:
                return self.map_cache[cache_key]

        # If not in cache or force_refresh is True, generate the map
        start_time = time.time()
        result = self.get_ranked_tags_map_uncached(
            chat_fnames, other_fnames, max_map_tokens, mentioned_fnames, mentioned_idents
        )
        end_time = time.time()
        self.map_processing_time = end_time - start_time

        # Store the result in the cache
        self.map_cache[cache_key] = result
        self.last_map = result

        return result

    def get_ranked_tags_map_uncached(
        self,
        chat_fnames,
        other_fnames=None,
        max_map_tokens=None,
        mentioned_fnames=None,
        mentioned_idents=None,
    ):
        """
        获取未缓存的排名标签映射。

        该方法用于生成一个包含文件和标签的树形结构，通过二分查找来优化树的大小，使其满足令牌数限制。

        参数:
            chat_fnames: 聊天中涉及的文件名列表
            other_fnames: 其他文件名列表，默认为None
            max_map_tokens: 最大映射令牌数，默认为None
            mentioned_fnames: 提到的文件名集合，默认为None
            mentioned_idents: 提到的标识符集合，默认为None

        返回:
            best_tree: 优化后的树形结构
        """
        # 初始化参数的默认值
        if not other_fnames:
            other_fnames = list()
        if not max_map_tokens:
            max_map_tokens = self.max_map_tokens
        if not mentioned_fnames:
            mentioned_fnames = set()
        if not mentioned_idents:
            mentioned_idents = set()

        # 创建进度显示器
        spin = Spinner("Updating repo map")

        # 获取排名的标签
        ranked_tags = self.get_ranked_tags(
            chat_fnames,
            other_fnames,
            mentioned_fnames,
            mentioned_idents,
            progress=spin.step,
        )

        # 处理特殊文件名
        other_rel_fnames = sorted(set(self.get_rel_fname(fname) for fname in other_fnames))
        special_fnames = filter_important_files(other_rel_fnames)
        ranked_tags_fnames = set(tag[0] for tag in ranked_tags)
        special_fnames = [fn for fn in special_fnames if fn not in ranked_tags_fnames]
        special_fnames = [(fn,) for fn in special_fnames]

        # 合并特殊文件名和排名标签
        ranked_tags = special_fnames + ranked_tags

        spin.step()

        # 初始化二分查找的变量
        num_tags = len(ranked_tags)
        lower_bound = 0
        upper_bound = num_tags
        best_tree = None
        best_tree_tokens = 0

        # 获取聊天相关文件名
        chat_rel_fnames = set(self.get_rel_fname(fname) for fname in chat_fnames)

        # 清空树缓存
        self.tree_cache = dict()

        # 设置初始中间值
        middle = min(max_map_tokens // 25, num_tags)
        
        # 使用二分查找找到最佳的树大小
        while lower_bound <= upper_bound:
            spin.step()

            # 生成当前大小的树
            tree = self.to_tree(ranked_tags[:middle], chat_rel_fnames)
            num_tokens = self.token_count(tree)

            # 计算误差百分比
            pct_err = abs(num_tokens - max_map_tokens) / max_map_tokens
            ok_err = 0.15
            
            # 更新最佳树
            if (num_tokens <= max_map_tokens and num_tokens > best_tree_tokens) or pct_err < ok_err:
                best_tree = tree
                best_tree_tokens = num_tokens

                if pct_err < ok_err:
                    break

            # 调整二分查找的边界
            if num_tokens < max_map_tokens:
                lower_bound = middle + 1
            else:
                upper_bound = middle - 1

            middle = (lower_bound + upper_bound) // 2

        # 结束进度显示
        spin.end()
        return best_tree
    tree_cache = dict()

    def render_tree(self, abs_fname, rel_fname, lois):
        """
        渲染文件树的方法。

        参数:
            abs_fname: 文件的绝对路径
            rel_fname: 文件的相对路径 
            lois: 感兴趣的行号列表(lines of interest)

        返回:
            格式化后的文件树字符串

        工作流程:
        1. 获取文件的修改时间作为缓存key的一部分
        2. 如果缓存中存在对应的渲染结果则直接返回
        3. 如果文件内容有更新(通过mtime判断):
           - 读取文件内容
           - 创建新的TreeContext对象并缓存
        4. 使用TreeContext处理感兴趣的行
        5. 格式化输出并缓存结果
        """
        mtime = self.get_mtime(abs_fname)
        key = (rel_fname, tuple(sorted(lois)), mtime)

        if key in self.tree_cache:
            return self.tree_cache[key]

        if (
            rel_fname not in self.tree_context_cache
            or self.tree_context_cache[rel_fname]["mtime"] != mtime
        ):
            code = self.io.read_text(abs_fname) or ""
            if not code.endswith("\n"):
                code += "\n"

            context = TreeContext(
                rel_fname,
                code,
                color=False,
                line_number=False,
                child_context=False,
                last_line=False,
                margin=0,
                mark_lois=False,
                loi_pad=0,
                # header_max=30,
                show_top_of_file_parent_scope=False,
            )
            self.tree_context_cache[rel_fname] = {"context": context, "mtime": mtime}

        context = self.tree_context_cache[rel_fname]["context"]
        context.lines_of_interest = set()
        context.add_lines_of_interest(lois)
        context.add_context()
        res = context.format()
        self.tree_cache[key] = res
        return res
    def to_tree(self, tags, chat_rel_fnames):
        if not tags:
            return ""

        cur_fname = None
        cur_abs_fname = None
        lois = None
        output = ""

        # add a bogus tag at the end so we trip the this_fname != cur_fname...
        dummy_tag = (None,)
        for tag in sorted(tags) + [dummy_tag]:
            this_rel_fname = tag[0]
            if this_rel_fname in chat_rel_fnames:
                continue

            # ... here ... to output the final real entry in the list
            if this_rel_fname != cur_fname:
                if lois is not None:
                    output += "\n"
                    output += cur_fname + ":\n"
                    output += self.render_tree(cur_abs_fname, cur_fname, lois)
                    lois = None
                elif cur_fname:
                    output += "\n" + cur_fname + "\n"
                if type(tag) is Tag:
                    lois = []
                    cur_abs_fname = tag.fname
                cur_fname = this_rel_fname

            if lois is not None:
                lois.append(tag.line)

        # truncate long lines, in case we get minified js or something else crazy
        output = "\n".join([line[:100] for line in output.splitlines()]) + "\n"

        return output


def find_src_files(directory):
    if not os.path.isdir(directory):
        return [directory]

    src_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            src_files.append(os.path.join(root, file))
    return src_files


def get_random_color():
    hue = random.random()
    r, g, b = [int(x * 255) for x in colorsys.hsv_to_rgb(hue, 1, 0.75)]
    res = f"#{r:02x}{g:02x}{b:02x}"
    return res


def get_scm_fname(lang):
    # Load the tags queries
    try:
        return resources.files(__package__).joinpath("queries", f"tree-sitter-{lang}-tags.scm")
    except KeyError:
        return


def get_supported_languages_md():
    from grep_ast.parsers import PARSERS

    res = """
| Language | File extension | Repo map | Linter |
|:--------:|:--------------:|:--------:|:------:|
"""
    data = sorted((lang, ex) for ex, lang in PARSERS.items())

    for lang, ext in data:
        fn = get_scm_fname(lang)
        repo_map = "✓" if Path(fn).exists() else ""
        linter_support = "✓"
        res += f"| {lang:20} | {ext:20} | {repo_map:^8} | {linter_support:^6} |\n"

    res += "\n"

    return res


if __name__ == "__main__":
    fnames = sys.argv[1:]

    chat_fnames = []
    other_fnames = []
    for fname in sys.argv[1:]:
        if Path(fname).is_dir():
            chat_fnames += find_src_files(fname)
        else:
            chat_fnames.append(fname)

    rm = RepoMap(root=".")
    repo_map = rm.get_ranked_tags_map(chat_fnames, other_fnames)

    dump(len(repo_map))
    print(repo_map)
