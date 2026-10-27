url = "url"

loader = WebBaseLoader(url)

docs = loader.load()

# this is for web page URLS.
# mostly works well for static pages , for dynamic pages meanong pages with heavy JS , use SeleniumBaseURL loader
