# 使用同步方式编写异步功能
import time
import asyncio

"""
我们都知道，yield在生成器中有中断的功能，可以传出值，也可以从函数外部接收值，而yield from的实现就是简化了yield操作。

在这个例子中yield titles返回了titles完整列表，而yield from titles实际等价于：
for title in titles:　# 等价于yield from titles
    yield title　

而yield from功能还不止于此，它还有一个主要的功能是省去了很多异常的处理，不再需要我们手动编写，其内部已经实现大部分异常处理。
"""


def generator_1(titles):
    yield titles


def generator_2(titles):
    yield from titles


titles = ['Python', 'Java', 'C++']
for title in generator_1(titles):
    print('生成器1:', title)
for title in generator_2(titles):
    print('生成器2:', title)


def generator_3():
    total = 0
    while True:
        x = yield
        print('加', x)
        if not x:
            break
        total += x
    return total


def generator_4():  # 委托生成器
    while True:
        total = yield from generator_3()  # 子生成器
        print('加和总数是:', total)


def main():  # 调用方
    """
    对于生成器g1，在最后传入None后，程序退出，报StopIteration异常并返回了最后total值是５。

    如果把g1.send()那５行注释掉，解注下面的g2.send()代码，则结果如下。可见yield from封装了处理常见异常的代码。
    对于g2即便传入None也不报异常，其中total = yield from generator_1()返回给total的值是generator_1()最终的return total

    【子生成器】：yield from后的generator_3()生成器函数是子生成器
    【委托生成器】：generator_4()是程序中的委托生成器，它负责委托子生成器完成具体任务。
    【调用方】：main()是程序中的调用方，负责调用委托生成器。

    yield from在其中还有一个关键的作用是：建立调用方和子生成器的通道，
    :return:
    """
    # g1 = generator_3()
    # g1.send(None)
    # g1.send(2)
    # g1.send(3)
    # g1.send(None)
    g2 = generator_4()
    g2.send(None)
    g2.send(2)
    g2.send(3)
    g2.send(None)


main()


@asyncio.coroutine  # 标志协程的装饰器
def taskIO_1():
    print('开始运行IO任务1...')
    yield from asyncio.sleep(2)  # 假设该任务耗时2s
    print('IO任务1已完成，耗时2s')
    return taskIO_1.__name__


@asyncio.coroutine  # 标志协程的装饰器
def taskIO_2():
    print('开始运行IO任务2...')
    yield from asyncio.sleep(3)  # 假设该任务耗时3s
    print('IO任务2已完成，耗时3s')
    return taskIO_2.__name__


@asyncio.coroutine  # 标志协程的装饰器
def main():  # 调用方
    tasks = [taskIO_1(), taskIO_2()]  # 把所有任务添加到task中
    done, pending = yield from asyncio.wait(tasks)  # 子生成器
    print(repr(pending))

    for r in done:  # done和pending都是一个任务，所以返回结果需要逐个调用result()
        print('协程无序返回值：' + r.result())


if __name__ == '__main__':
    start = time.time()
    loop = asyncio.get_event_loop()  # 创建一个事件循环对象loop
    try:
        loop.run_until_complete(main())  # 完成事件循环，直到最后一个任务结束
    finally:
        loop.close()  # 结束事件循环
    print('所有IO任务总耗时%.5f秒' % float(time.time() - start))
