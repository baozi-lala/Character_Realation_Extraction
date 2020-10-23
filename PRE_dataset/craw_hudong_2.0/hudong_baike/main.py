from scrapy import cmdline

if __name__ == '__main__':
    cmdline.execute("scrapy crawl hudong_baike -s JOBDIR=paths_to_somewhere".split())