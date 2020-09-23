import asyncio
import time

"""
在Python 3.4中，我们发现很容易将协程和生成器混淆(虽然协程底层就是用生成器实现的)，所以在后期加入了其他标识来区别协程和生成器。
在Python 3.5开始引入了新的语法async和await，以简化并更好地标识异步IO。

要使用新的语法，只需要做两步简单的替换：
　　把@asyncio.coroutine替换为async；
　　把yield from替换为await。
"""

async def taskIO_1():
    print('开始运行IO任务1...')
    await asyncio.sleep(2)  # 假设该任务耗时2s
    print('IO任务1已完成，耗时2s')
    return taskIO_1.__name__


async def taskIO_2():
    print('开始运行IO任务2...')
    await asyncio.sleep(3)  # 假设该任务耗时3s
    print('IO任务2已完成，耗时3s')
    return taskIO_2.__name__


async def main():  # 调用方
    tasks = [taskIO_1(), taskIO_2()]  # 把所有任务添加到task中
    done, pending = await asyncio.wait(tasks)  # 子生成器
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
