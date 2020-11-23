from scrapy import cmdline
#  -s JOBDIR=paths_to_somewhere
if __name__ == '__main__':
    cmdline.execute("scrap crawl eventspider".split())