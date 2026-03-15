from icrawler.builtin import BingImageCrawler # Bing පමණක් Import කරමු
import os

# පින්තූර සේව් වන ස්ථානය
base_dir = 'dataset/binary_classifier/non_rice_leaf'

if not os.path.exists(base_dir):
    os.makedirs(base_dir)

# අපිට අවශ්‍ය Keywords
keywords = [
    "Barnyard grass paddy weed", 
    "Ischaemum rugosum rice weed",
    "Cyperus iria paddy field",
    "laptop keyboard keys",
    "human fingers and palm",
    "mango tree leaves",
    "wild grass on ground"
]

for kw in keywords:
    print(f"Downloading images for: {kw}")
    # මෙතන BingImageCrawler ලෙස නිවැරදිව සඳහන් කර ඇත
    bing_crawler = BingImageCrawler(storage={'root_dir': base_dir})
    # හැම keyword එකකටම පින්තූර 50 බැගින්
    bing_crawler.crawl(keyword=kw, max_num=50)

print("\nDownload completed!")