from scrapy import cmdline
# -s JOBDIR=paths_to_somewhere
if __name__ == '__main__':
    cmdline.execute("scrapy crawl hudong_baike".split())