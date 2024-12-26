from aider.repomap import RepoMap
from aider.utils import Spinner
from collections import defaultdict, Counter
from pathlib import Path
import math
from tqdm import tqdm
from aider.special import filter_important_files

class RepoMap_Unit(RepoMap):
    """
    A unit test version of RepoMap that inherits from the base RepoMap class.
    This class can be used for testing RepoMap functionality in isolation.
    """

    def __init__(
        self,
        map_tokens=1024,
        root=None,
        main_model=None,
        io=None,
        repo_content_prefix=None,
        verbose=False,
        max_context_window=None,
        map_mul_no_files=1.5,
        refresh=False,
    ):
        super().__init__(
            map_tokens=map_tokens,
            root=root,
            main_model=main_model,
            io=io,
            repo_content_prefix=repo_content_prefix,
            verbose=verbose,
            max_context_window=max_context_window,
            map_mul_no_files=map_mul_no_files,
            refresh=refresh,
        )

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
        ranked_tags = self.get_ranked_tags_unit(
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

    def get_ranked_tags_unit(
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
            if ident == "toString" or ident == "equals" or ident == "hashCode":
                continue

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

        distances = nx.single_source_dijkstra_path_length(G, next(iter(chat_rel_fnames)), weight='weight')
        # 按距离排序
        sorted_dist_nodes = sorted(distances.items(), key=lambda x: x[1])
        sorted_ranked_tags = sorted(ranked_tags, key=lambda x: distances.get(x.rel_fname, float('inf')))
        ranked_tags = sorted_ranked_tags

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
