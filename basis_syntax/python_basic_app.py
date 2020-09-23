import array
import unittest
from collections import namedtuple
from dis import dis
from inspect import signature
from typing import Union


def sorted_key(k):
    return len(k)


def tag(name, *content, cls=None, **attrs):
    """生成一个或多个HTML标签"""
    if cls is not None:
        attrs['class'] = cls
    if attrs:
        attr_str = ''.join(' %s="%s"' % (attr, value) for attr, value in sorted(attrs.items()))
    else:
        attr_str = ''
    if content:
        return '\n'.join('<%s%s>%s</%s>' % (name, attr_str, c, name) for c in content)
    else:
        return '<%s%s/>' % (name, attr_str)


def f(a, c=2, *, b=1):
    """ 如果不想支持数量不定的定位参数，但是想支持仅限关键字参数，在签名中放一个*即可"""
    return a, b


def clip(text, max_len=80):
    """在max_len前面或后面的第一个空格处截断文本"""
    end = None
    if len(text) > max_len:
        space_before = text.rfind(' ', 0, max_len)
        if space_before >= 0:
            end = space_before
        else:
            space_after = text.rfind(' ', max_len)
            if space_after >= 0:
                end = space_after
    if end is None:
        end = len(text)
    return text[:end].rstrip()


class C(object):
    """
    使用property装饰类可以在实例变量赋值时候校验参数的正确性，见https://www.programiz.com/python-programming/property
    """

    def __init__(self, value=0):
        self._x = value

    @property
    def x(self):
        print("I am the 'x' property.")
        return self._x

    @x.setter
    def x(self, value):
        if value >= 100:
            raise ValueError('value gt 100 is not possible')
        self._x = value

    @x.deleter
    def x(self):
        del self._x


class AppTest(unittest.TestCase):

    def setUp(self) -> None:  # ​Python 3 新特性：类型注解
        print("setup")

    def test_properties(self):
        c_v = C()
        c_v.x = 1
        c_v.y = 1
        print(c_v.x)
        print(c_v._x)

    def test_function_inspect(self):
        """
        使用inspect模块来提取函数的签名
        inspect.signature函数返回一个inspect.Signature对象，他有一个parameters属性，它是一个有序映射，把参数名和inspect.Paramenter
        对应起来，各个Parameter属性有自己的属性，如name、default、kind。考虑到None是有效的默认值，使用特殊的inspect._empty值表示没有默认值

        kind属性值是_ParameterKind类的5个值之一：
            POSITIONAL_OR_KEYWORD: 可以通过定位参数和关键字参数传入的形参（多数python函数的参数属于此类）
            VAR_POSITIONAL: 定位参数元组
            VAR_KEYWORD: 关键字参数字典
            KEYWORD_ONLY: 仅限关键字参数（python 3新增）
            POSITIONAL_ONLY: 仅限定位参数，目前python声明的语法不支持，有些c语言的实现且不接受关键字参数的函数（divmod)支持

        除了name、default、kind，inspect.Parameter对象还有一个annotation（注解）属性，它的值通常是inspect._empty，但是可能包含
        python3新的注解句法提供的函数签名元数据

      :return:
        """

        sig = signature(clip)  # 对函数clip内省
        print(str(sig))  # (text, max_len=80)
        for name, param in sig.parameters.items():
            print(param.kind, ':', name, '=', param.default)

    def test_inspect_bind(self):
        """
        inpect.Signature对象有个bind方法，它可以把任意个参数绑定到签名中的形参上，所用的规则和实参到形参的匹配方式一样。框架可以使用这个方法
        在真正调用函数前验证参数
        :return:
        """
        sig = signature(tag)
        my_tags = {'name': 'img', 'title': 'Sunset Boulevard', 'src': 'sunset.jpg', 'cls': 'framed'}
        bound_args = sig.bind(**my_tags)
        for name, value in bound_args.arguments.items():
            print(name, '=', value)

        del my_tags['name']
        # bound_args = sig.bind(**my_tags)   # 失败，抛出TypeError，缺少name参数

    def test_function_meta_data(self):
        """
        查看函数对象的元信息
        函数对象有一个__default__属性，它的值是一个元组，里面保存着定位参数和关键字参数的默认值。仅限关键字参数的默认值在
        __kwdefaults__属性中。参数的名称在__code__属性中，它的值是一个code对象引用，自身也有很多属性
        """
        print(f.__defaults__)
        print(f.__kwdefaults__)
        print(clip.__defaults__)
        print(clip.__kwdefaults__)
        print(clip.__code__)
        # 参数名称在__code__.co_varnames中，但是里面也有函数定义体中创建的局部变量，因此参数名称是前n个字符串，N的值由__code__.co_argcunt
        # 确定。这里不包含前缀为*或**的变长参数。
        print(clip.__code__.co_varnames)
        print(clip.__code__.co_argcount)  # 参数的个数

        # 参数的默认值只能通过它们在__default__元组中的位置确定，因此要从后向前扫描才能把参数和默认值对应起来
        # 在这个示例中，clip函数有两个参数，text和max_len，其中一个有默认值，为80。因此它必然属于最后一个参数，即max_len，
        # 这中内省的方式比较麻烦，可以使用inspect模块来更方便的提取函数的相关信息。

    def test_tag(self):
        print(tag('br'))
        print(tag('p', 'hello'))
        print(tag('p', 'hello', 'world'))
        print(tag('p', 'hello', 'world', id=2))
        print(tag('p', 'hello', 'world', cls='sidebar'))

        my_tags = {'name': 'img', 'title': 'Sunset Boulevard', 'src': 'sunset.jpg', 'cls': 'framed'}
        # 在my_tags前面加上**，字典中的所有元素作为单个参数传入，同名键会绑定到对应的具名参数上，余下的则会被**attrs捕获
        # 存入一个字典
        print(tag(**my_tags))

    def test_restrict_keyword_parm(self):
        """
        仅限关键字参数是python3新增的特性，如tag方法中的cls参数只能通过关键字参数限定，它一定不会捕获未命名的定位参数。定义函数时
        若想指定尽限关键字参数，要把他们都当道前面有*的参数后面。
        如果不想支持数量不定的定位参数，但是想支持仅限关键字参数，在签名中放一个*即可。
        :return:
        """
        print(f(1, b=2))

    def test_assembly(self):
        """
        查看python语言的汇编码
        :return:
        """
        dis('{1}')  # 查看实例化集合的汇编码
        dis('set([1])')  # 查看实例化集合的汇编码

    def test_set(self):
        """
        给定两个集合a和b，a | b返回的是它们的合集，a & b得到的是交集，而a-b得到的是差集。
        :return:
        """
        print(set([1, 2, 3]) | set([2, 3, 4]))
        # 集合字面量使用{}来表示，但是空集合必须使用set()，否则{}表示空字典类型
        print({'1', '2'})
        print(type({}))  # <class 'dict'>
        print(set())

        # frozenset使用
        frozenset(range(0, 10))
        # 集合推导
        set1 = {i * 2 for i in range(1, 10)}
        print(set1)
        print(type(set1))

        # 集合的比较运算
        set2 = {2, 4}
        print(set2 <= set1)  # 判断set2是否是set1的子集
        print(set2 < set1)  # 判断set2是否是set1的真子集
        print(set2 > set1)  # 判断set2是否是set1的真父集
        print(set2 >= set1)  # 判断set2是否是set1的父集

    def test_memory_view(self):
        """
        内存视图：memoryview是一个内置类，它能让用户在不复制内容的情况下操作同一个数组的不同切片。memoryview的概念受到了NumPy的启发
        内存视图其实是泛化和去数学化的NumPy数组。它让你在不需要复制内容的前提下，在数据结构之间共享内存。其中数据结构可以是任何形式，
        比如PIL图片、SQLite数据库和NumPy的数组
        :return:
        """
        arr = array.array('h', [-2, -1, 0, 1, 2])
        memv = memoryview(arr)
        print(len(memv))
        print(memv[1])
        print(memv.tolist())
        memv_oct = memv.cast('B')
        print(memv_oct.tolist())
        print(len(memv_oct))  # 10
        memv_oct[5] = 4  # 因为我们把占2个字节的整数的高位字节改成了4，所以这个有符号整数的值就变成了1024。
        print(arr)

    def test_sorted(self) -> object:
        print(sorted(['1', '3222', '22'], key=sorted_key))

    def test_for_comprehension(self):
        """
        for 推导
        :return:
        """
        arr = [x ** 2 for x in range(1, 10)]
        print(arr)

    def test_generate_expression(self):
        """
        生成器表达式
        :return:
        """
        t = tuple(x for x in '1234')
        print(t)

        arr = array.array('I', (x ** 2 for x in range(1, 10)))  # 用生成器表达式初始化元组和数组
        print(arr)

        # 使用生成器表达式计算笛卡尔积
        for t_shirt in ('%s %s' % (c, s) for c in ['black', 'yellow'] for s in ['S', 'M', 'L']):
            print(t_shirt)

    def test_tuple_unpack(self):
        """
        元组拆分
        :return:
        """
        k, v = ('1', 2)
        # 交换变量
        v, k = k, v
        print(k)
        print(v)

        for k, _ in [('1', 2), ('2', 3)]:
            print(k)

    def test_named_tuple(self):
        """
        具名元组：继承自普通元组，有_fields类属性、类方法：_make(iterable)和实例方法:_asdict()
        :return:
        """
        Card = namedtuple('Card', ['rank', 'suit'])
        a_card = Card('1', 'suit-1')
        b_card = Card('2', 'suit-2')
        c_card = Card._make(['a', 'c'])  # 通过接受一个可迭代对象生成这个类的实例，作用和Card(*['a','c'])一样
        d_card = Card(*['a', 'c'])
        print(d_card._asdict())  # 把具名元组以collections.OrderedDict的形式返回

        for (kv, v) in d_card._asdict().items():
            print(kv + ':' + str(v))

        print(a_card)
        print(b_card)
        print(c_card)
        print(d_card)

    @staticmethod
    def test_if_condition():
        c = [1, 2]
        # 三目表达式： expr1 if condition1 else expr2; 如果condition1满足条件，则执行expr1,否则执行expr2
        callbacks = [1] if c is None else c
        print(callbacks)  # 输出[1, 2]

    @staticmethod
    def union_type_test(param: Union[int, str]):
        print(type(param))
        print(param & 1)
