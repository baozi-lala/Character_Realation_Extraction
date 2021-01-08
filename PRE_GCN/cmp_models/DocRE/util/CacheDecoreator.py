import pandas as pd
import time
import os


class CacheDecoreator(object):
    def __init__(self, base_dir, task_name):
        self.base_dir = base_dir
        self.task_name = task_name

    def category(self, cate, force=False):
        def real_func(func):
            def wrapper(*args, **kwargs):
                import os
                cache_name = os.path.join(
                    self.base_dir, cate, self.task_name + '.pkl')
                if not force and os.path.exists(cache_name):
                    return pd.read_pickle(cache_name)
                else:
                    import time
                    start_time = time.time()
                    result = func(*args, **kwargs)
                    end_time = time.time()
                    pd.to_pickle(result, cache_name)
                    pd.to_pickle(
                        {
                            'start_time': start_time,
                            'end_time': end_time,
                            'duration': end_time - start_time
                        },
                        os.path.join(
                            self.base_dir, cate, self.task_name + '.time.pkl'))
                return result

            return wrapper

        return real_func

    def read_cache(self, cate):
        return pd.read_pickle(os.path.join(self.base_dir, cate,
                                           self.task_name + '.pkl'))

    def read_time(self, cate):
        return pd.read_pickle(os.path.join(self.base_dir, cate,
                                           self.task_name + '.time.pkl'))


class TimeDecoreator(object):
    def __init__(self):
        pass

    def category(self):
        def real_func(func):
            def wrapper(*args, **kwargs):
                import time
                start_time = time.time()
                result = func(*args, **kwargs)
                end_time = time.time()
                print("run time:", (end_time - start_time))
                return result

            return wrapper

        return real_func


cache = CacheDecoreator('..\\log', "test")


@cache.category('timelog')
def exactly_string_matching_minimum_cache(name):
    return 0


# esmm = exactly_string_matching_minimum_cache("test")


@cache.category('timelog', force=True)  # 强制运行
def exactly_string_matching_minimum_cache(name):
    return 0


# esmm = exactly_string_matching_minimum_cache("test")
