import os
import javalang
from pathlib import Path
from typing import Dict, Set, Tuple, Optional
from dataclasses import dataclass, field

# 在文件顶部定义常量
JAVA_LANG_TYPES = {
    'String', 'Integer', 'Long', 'Boolean', 'Double', 'Float', 
    'Byte', 'Short', 'Character', 'Object', 'Class', 'System', 
    'Thread', 'Runnable', 'Exception', 'RuntimeException'
}

JAVA_UTIL_TYPES = {
    'List', 'Map', 'Set', 'Collection', 'Iterator', 'ArrayList', 
    'HashMap', 'HashSet', 'Properties'
}

PRIMITIVE_TYPES = {
    'void', 'int', 'long', 'boolean', 'double', 'float', 'byte', 'short', 'char'
}

@dataclass
class MethodCall:
    class_name: str
    method_name: str

    def __str__(self):
        return f"{self.class_name}.{self.method_name}"

@dataclass
class ClassDependency:
    name: str
    file_path: str = ""
    imports: Set[str] = field(default_factory=set)
    field_types: Set[str] = field(default_factory=set)
    method_calls: Dict[str, Set[MethodCall]] = field(default_factory=dict)
    implements: Set[str] = field(default_factory=set)
    extends: Optional[str] = None
    # 新增：记录依赖类的使用情况
    usage_details = {}  # 格式: {'类名': {'fields': set(), 'methods': set()}}

    def __init__(self, name: str):
        self.name = name
        self.file_path = None
        self.extends = None
        self.implements = []
        self.field_types = []
        # 新增：记录依赖类的使用情况
        self.usage_details = {}  # 格式: {'类名': {'fields': set(), 'methods': set()}}

class JavaDependencyAnalyzer:
    def __init__(self, base_path: str, root_dir: str = None):
        self.base_path = base_path
        self.root_dir = root_dir or os.path.abspath(os.path.join(base_path, "..", "..", ".."))
        self.class_to_file = {}  # 类名到文件路径的映射
        self.package_to_classes = {}  # 包名到类名的映射

    def _scan_java_files(self):
        """扫描并建立类名到文件路径的映射"""
        for root, dirs, files in os.walk(self.root_dir):
            for file in files:
                if file.endswith('.java'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            source = f.read()
                            tree = javalang.parse.parse(source)
                            package = tree.package.name if tree.package else ""
                            
                            # 记录包中的类
                            if package not in self.package_to_classes:
                                self.package_to_classes[package] = set()
                            
                            # 直接从编译单元的类型列表中获取类和接口
                            for node in tree.types:
                                if isinstance(node, (javalang.tree.ClassDeclaration, javalang.tree.InterfaceDeclaration)):
                                    full_name = f"{package}.{node.name}" if package else node.name
                                    normalized_path = os.path.normpath(file_path)
                                    self.class_to_file[node.name] = (normalized_path, full_name)
                                    self.class_to_file[full_name] = (normalized_path, full_name)
                                    self.package_to_classes[package].add(node.name)
                    except Exception as e:
                        continue

    def _resolve_type(self, type_name: str, imports: Set[str], current_package: str) -> Optional[str]:
        """解析完整的类型名称"""
        if type_name in self.class_to_file:
            return type_name
        
        # 检查是否是当前包中的类
        if current_package:
            full_name = f"{current_package}.{type_name}"
            if full_name in self.class_to_file:
                return full_name
        
        # 检查导入
        for imp in imports:
            if imp.endswith(f".{type_name}"):
                return imp
            
            # 检查通配符导入
            if imp.endswith(".*"):
                package = imp[:-2]
                if package in self.package_to_classes and type_name in self.package_to_classes[package]:
                    return f"{package}.{type_name}"
        
        return None

    def _analyze_class(self, node, package: str, imports: Set[str]) -> ClassDependency:
        """分析类的依赖关系"""
        # 缓存类型解析结果
        type_cache = {}
        
        def resolve_type(type_name: str) -> str:
            """解析类型的完整名称（带缓存）"""
            if type_name in type_cache:
                return type_cache[type_name]
            
            result = type_name
            
            # 如果已经是完整包名，直接返回
            if '.' not in type_name:
                # 如果是同包的类
                if package and type_name in self.package_to_classes.get(package, set()):
                    result = f"{package}.{type_name}"
                # 如果在导入映射中找到
                elif type_name in type_mapping:
                    result = type_mapping[type_name]
                # 如果是通过通配符导入的类
                else:
                    for pkg in wildcard_packages:
                        if type_name in self.package_to_classes.get(pkg, set()):
                            result = f"{pkg}.{type_name}"
                            break
                    else:
                        # 如果是 java.lang 包中的类
                        if type_name in JAVA_LANG_TYPES:
                            result = f"java.lang.{type_name}"
                        # 如果是 java.util 包中的类
                        elif type_name in JAVA_UTIL_TYPES:
                            result = f"java.util.{type_name}"
            
            type_cache[type_name] = result
            return result
        
        class_name = f"{package}.{node.name}" if package else node.name
        dependency = ClassDependency(class_name)
        
        # 收集所有使用的类型
        used_types = set()
        
        # 处理通配符导入
        wildcard_packages = {imp[:-2] for imp in imports if imp.endswith('.*')}
        
        # 建立简单类名到完整包名的映射
        type_mapping = {}
        for imp in imports:
            if not imp.endswith('.*'):
                simple_name = imp.split('.')[-1]
                type_mapping[simple_name] = imp
        
        def add_usage(type_name: str, usage_type: str, member_name: str):
            """记录类型的使用情况"""
            resolved_type = resolve_type(type_name)
            if resolved_type not in dependency.usage_details:
                dependency.usage_details[resolved_type] = {'fields': set(), 'methods': set()}
            dependency.usage_details[resolved_type][usage_type].add(member_name)
        
        # 处理继承
        if hasattr(node, 'extends') and node.extends:
            if isinstance(node.extends, list):
                # 接口可以继承多个接口
                dependency.extends = [ext.name for ext in node.extends]
                for ext in dependency.extends:
                    used_types.add(resolve_type(ext))
            else:
                # 类只能继承一个类
                ext_name = node.extends.name
                used_types.add(resolve_type(ext_name))
                dependency.extends = ext_name
        
        # 处理接口实现
        if hasattr(node, 'implements') and node.implements:
            try:
                dependency.implements = [impl.name for impl in node.implements]
                for impl in dependency.implements:
                    used_types.add(resolve_type(impl))
            except (AttributeError, TypeError):
                dependency.implements = []
        
        # 处理字段类型
        if hasattr(node, 'fields') and node.fields:
            for field in node.fields:
                if hasattr(field.type, 'name'):
                    type_name = field.type.name
                    used_types.add(resolve_type(type_name))
                    # 记录字段使用
                    for declarator in field.declarators:
                        add_usage(type_name, 'fields', declarator.name)
        
        # 处理方法
        if hasattr(node, 'methods') and node.methods:
            for method in node.methods:
                # 返回类型
                if hasattr(method, 'return_type') and method.return_type:
                    if hasattr(method.return_type, 'name'):
                        used_types.add(resolve_type(method.return_type.name))
                
                # 参数类型
                if hasattr(method, 'parameters') and method.parameters:
                    for param in method.parameters:
                        if hasattr(param.type, 'name'):
                            used_types.add(resolve_type(param.type.name))
                
                # 方法体中的类型引用
                if hasattr(method, 'body') and method.body:
                    for statement in method.body:
                        # 处理方法调用
                        for path, invocation in statement.filter(javalang.tree.MethodInvocation):
                            if hasattr(invocation, 'qualifier') and invocation.qualifier:
                                qualifier = invocation.qualifier
                                if qualifier[0].isupper():
                                    used_types.add(resolve_type(qualifier))
                                    # 记录方法调用
                                    add_usage(qualifier, 'methods', invocation.member)
        
        # 从导入语句中收集依赖
        for imp in imports:
            # 只添加具体的类导入，不添加.*导入
            if not imp.endswith('.*'):
                used_types.add(imp)
        
        # 更新字段类型列表
        dependency.field_types = list(used_types)
        
        return dependency

    def analyze_with_depth(self, initial_file: str, level: int = 1) -> Dict[int, Dict[str, ClassDependency]]:
        """分析Java文件及其依赖的类"""
        print(f"\nProcessing file at depth 0: {initial_file}")
        
        # 规范化初始文件路径
        initial_file = os.path.normpath(initial_file)
        
        # 先扫描所有Java文件
        self._scan_java_files()
        print(f"Found {len(self.class_to_file)} classes in total")
        
        # 初始化每个层级的依赖字典
        dependencies_by_level = {i: {} for i in range(level)}
        analyzed_files = set()
        files_to_analyze = {(initial_file, 0)}
        
        while files_to_analyze:
            file_path, current_depth = files_to_analyze.pop()
            if file_path in analyzed_files or current_depth >= level:
                continue
            
            analyzed_files.add(file_path)
            print(f"Analyzing depth {current_depth}: {file_path}")
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    source = f.read()
                    print(f"Successfully read file: {file_path}")
                    
                    tree = javalang.parse.parse(source)
                    package = tree.package.name if tree.package else ""
                    print(f"Found package: {package}")
                    
                    # 收集导入
                    imports = {imp.path for imp in tree.imports if imp.path}
                    print("Found imports:")
                    for imp in imports:
                        print(f"  - {imp}")
                    
                    # 分析类和接口
                    for node in tree.types:
                        if isinstance(node, (javalang.tree.ClassDeclaration, javalang.tree.InterfaceDeclaration)):
                            print(f"\nAnalyzing class/interface: {node.name}")
                            dependency = self._analyze_class(node, package, imports)
                            dependency.file_path = file_path
                            
                            # 添加到当前层级
                            dependencies_by_level[current_depth][dependency.name] = dependency
                            print(f"Added dependency at level {current_depth}: {dependency.name}")
                            
                            # 处理下一级依赖
                            if current_depth + 1 < level:
                                self._process_next_level_dependencies(
                                    dependency, current_depth, analyzed_files, files_to_analyze
                                )
            
            except Exception as e:
                print(f"Error analyzing {file_path}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        return dependencies_by_level

    def _process_next_level_dependencies(self, dependency, current_depth, analyzed_files, files_to_analyze):
        """处理下一级依赖"""
        # 处理继承的类
        if dependency.extends:
            next_file = self.class_to_file.get(dependency.extends, (None, None))[0]
            if next_file and next_file not in analyzed_files:
                files_to_analyze.add((next_file, current_depth + 1))
                print(f"Adding extended class for analysis: {next_file}")
        
        # 处理实现的接口
        for impl in dependency.implements:
            next_file = self.class_to_file.get(impl, (None, None))[0]
            if next_file and next_file not in analyzed_files:
                files_to_analyze.add((next_file, current_depth + 1))
                print(f"Adding implemented interface for analysis: {next_file}")
        
        # 处理字段类型
        for field_type in dependency.field_types:
            next_file = self.class_to_file.get(field_type, (None, None))[0]
            if next_file and next_file not in analyzed_files:
                files_to_analyze.add((next_file, current_depth + 1))
                print(f"Adding field type for analysis: {next_file}")

def print_dependencies(dependencies: Dict[int, Dict[str, ClassDependency]], verbose: bool = False):
    """打印依赖关系"""
    print("\nDetailed dependencies:")
    
    for level, deps in dependencies.items():
        if not deps:
            continue
            
        print(f"\n=== Level {level} Dependencies ({len(deps)} classes) ===\n")
        
        for class_name, dep in deps.items():
            print(f"类: {class_name}")
            # 打印相对路径
            if dep.file_path:
                print(f"文件: {dep.file_path}")
            
            
            if verbose:
                if dep.field_types:
                    print(f"\n使用的类型: ({len(dep.field_types)} dependencies)")
                    for type_name in sorted(dep.field_types):
                        print(f"  - {type_name}")
                        if type_name in dep.usage_details:
                            if dep.usage_details[type_name]['fields']:
                                print(f"    字段: {', '.join(sorted(dep.usage_details[type_name]['fields']))}")
                            if dep.usage_details[type_name]['methods']:
                                print(f"    方法: {', '.join(sorted(dep.usage_details[type_name]['methods']))}")

                if dep.extends:
                    print(f"\n继承:")
                    if isinstance(dep.extends, list):
                        for ext in dep.extends:
                            print(f"  - {ext}")
                    else:
                        print(f"  - {dep.extends}")
                
                if dep.implements:
                    print(f"\n实现接口:")
                    for impl in dep.implements:
                        print(f"  - {impl}")
            
            print()

def analyze_java_dependencies(file_path: str, root_dir: str = None, level: int = 1, verbose: bool = False):
    """分析Java文件的依赖关系"""
    analyzer = JavaDependencyAnalyzer(os.path.dirname(file_path), root_dir)
    dependencies = analyzer.analyze_with_depth(file_path, level)
    
    # 计算总依赖数
    total_deps = sum(len(deps[class_name].field_types) for deps in dependencies.values() for class_name in deps)
    total_classes = sum(len(deps) for deps in dependencies.values())
    
    print("\nAnalysis complete:")
    print(f"Found {total_deps} dependencies across {level} levels")
    print(f"Total classes analyzed: {total_classes}")
    
    for l in range(level):
        deps_at_level = dependencies[l]
        if deps_at_level:
            deps_count = sum(len(dep.field_types) for dep in deps_at_level.values())
            print(f"Level {l}: {len(deps_at_level)} classes with {deps_count} dependencies")
    
    print_dependencies(dependencies, verbose)

if __name__ == "__main__":
    # 测试 Company.java
    # file_path = "data/basic/test/basic/data/Company.java"
    # root_dir = "data/basic"  # 源代码根目录
    # analyze_java_dependencies(file_path, root_dir, level=2)

    # 测试 NacosNamingService.java
    file_path = "D:/dev/project/llm/llm_commenter/data/nacos/java/com/alibaba/nacos/client/naming/NacosNamingService.java"
    root_dir = "D:/dev/project/llm/llm_commenter/data/nacos/java"  # 源代码根目录
    analyze_java_dependencies(file_path, root_dir, level=2, verbose=False) 