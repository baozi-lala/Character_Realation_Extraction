from scrapy import cmdline

if __name__ == '__main__':
    cmdline.execute("scrapy crawl magi -s JOBDIR=paths_to_somewhere".split())