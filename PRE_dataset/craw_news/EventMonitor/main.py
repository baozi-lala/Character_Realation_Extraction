from scrapy import cmdline

if __name__ == '__main__':
    cmdline.execute("scrap y crawl eventspider -s JOBDIR=paths_to_somewhere".split())