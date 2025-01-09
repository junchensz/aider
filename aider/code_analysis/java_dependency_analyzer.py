import javalang
from dataclasses import dataclass
from typing import Set, Dict, List
from collections import defaultdict

@dataclass(frozen=True)
class MethodReference:
    """方法引用信息"""
    class_name: str
    method_name: str
    
    def __str__(self):
        return f"{self.class_name}.{self.method_name}"

@dataclass
class ClassDependency:
    """类依赖信息"""
    name: str
    imports: Set[str] = None  # 导入的包
    field_types: Set[str] = None  # 字段类型
    method_calls: Dict[str, Set[MethodReference]] = None  # 方法调用信息
    
    def __post_init__(self):
        self.imports = set() if self.imports is None else self.imports
        self.field_types = set() if self.field_types is None else self.field_types
        self.method_calls = defaultdict(set) if self.method_calls is None else self.method_calls

class JavaAnalyzer:
    def __init__(self, file_path: str):
        self.file_path = file_path
        with open(file_path, 'r', encoding='utf-8') as f:
            self.source = f.read()
        self.tree = javalang.parse.parse(self.source)
        
    def analyze(self) -> List[ClassDependency]:
        """分析Java文件中的所有类的依赖关系"""
        dependencies = []
        
        # 遍历所有类声明
        for path, class_node in self.tree.filter(javalang.tree.ClassDeclaration):
            dependency = ClassDependency(name=class_node.name)
            
            # 收集导入信息
            for imp in self.tree.imports:
                if imp.path:
                    dependency.imports.add(imp.path)
            
            # 收集字段类型
            for field in class_node.fields:
                if hasattr(field.type, 'name'):
                    dependency.field_types.add(field.type.name)
            
            # 分析方法调用
            self._analyze_method_calls(class_node, dependency)
            
            dependencies.append(dependency)
        
        return dependencies
    
    def _analyze_method_calls(self, class_node: javalang.tree.ClassDeclaration, 
                            dependency: ClassDependency):
        """分析类中的方法调用"""
        def get_field_type(field_name):
            """获取字段类型"""
            for field in class_node.fields:
                if any(d.name == field_name for d in field.declarators):
                    return field.type.name
            return None

        def analyze_expression(expr, current_method):
            """分析表达式中的方法调用"""
            if isinstance(expr, javalang.tree.This):
                if hasattr(expr, 'selectors') and expr.selectors:
                    current_type = None
                    
                    for selector in expr.selectors:
                        if isinstance(selector, javalang.tree.MemberReference):
                            # 处理 this.company
                            current_type = get_field_type(selector.member)
                        elif isinstance(selector, javalang.tree.MethodInvocation):
                            # 处理 getName() 调用
                            if current_type:
                                ref = MethodReference(
                                    class_name=current_type,
                                    method_name=selector.member
                                )
                                dependency.method_calls[current_method].add(ref)

        # 分析方法声明
        for method in class_node.methods:
            current_method = method.name
            
            if not method.body:
                continue
            
            # 分析方法体中的语句
            for statement in method.body:
                if isinstance(statement, javalang.tree.ReturnStatement):
                    if hasattr(statement, 'expression'):
                        expr = statement.expression
                        analyze_expression(expr, current_method)
                        
                        # 处理直接的方法调用
                        if isinstance(expr, javalang.tree.MethodInvocation):
                            if isinstance(expr.qualifier, javalang.tree.MemberReference):
                                field_type = get_field_type(expr.qualifier.member)
                                if field_type:
                                    ref = MethodReference(
                                        class_name=field_type,
                                        method_name=expr.member
                                    )
                                    dependency.method_calls[current_method].add(ref)

def analyze_java_file(file_path: str):
    """分析Java文件并打印依赖信息"""
    analyzer = JavaAnalyzer(file_path)
    dependencies = analyzer.analyze()
    
    for dep in dependencies:
        print(f"\n类: {dep.name}")
        
        if dep.imports:
            print("\n导入的包:")
            for imp in sorted(dep.imports):
                print(f"  - {imp}")
        
        if dep.field_types:
            print("\n使用的类型:")
            for type_name in sorted(dep.field_types):
                print(f"  - {type_name}")
        
        if any(calls for calls in dep.method_calls.values()):
            print("\n方法调用:")
            for method, calls in dep.method_calls.items():
                if calls:  # 只打印有调用的方法
                    print(f"\n  方法 {method} 调用了:")
                    for call in sorted(calls, key=lambda x: str(x)):
                        print(f"    - {call}")

if __name__ == "__main__":
    java_file = "data/Person.java"
    analyze_java_file(java_file) 