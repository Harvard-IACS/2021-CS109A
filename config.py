COURSE_NAME = 'CS109A'

AUTHOR = 'Pavlos Protopapas'

SEMESTER = 'Fall'

YEAR = '2021'

SITEURL = 'https://harvard-iacs.github.io/2021-CS109A'

GITHUB = 'https://github.com/Harvard-IACS/2021-CS109A'

COLOR = '#c90016'

# Define Navbar links
# ex. ('Link Name', 'URL')
MENUITEMS = [
    ('Syllabus', 'pages/syllabus.html'),
    ('Calendar', 'pages/calendar.html'),
    ('Schedule', 'pages/schedule.html'),
    #('Materials', 'pages/materials.html'),
    # ('Videos', 'pages/videos.html'),
    #('Modules', 'pages/modules.html'),
    # ('Projects', 'pages/projects.html'),
    ('FAQ', 'pages/faq.html'),
    ('Preparation', 'pages/preparation.html')
    # ('Resources', 'pages/resources.html')
    # ('Sections', 'category/sections.html')
]

# Uncomment following line if you want document-relative URLs when developing
RELATIVE_URLS = True

PATH = 'content'

OUTPUT_PATH = 'docs'

TIMEZONE = 'EST'

DEFAULT_LANG = 'en'

# ================================
# Pelican Settings
# Do not modify
# ================================

FEED_ALL_ATOM = None

CATEGORY_FEED_ATOM = None

TRANSLATION_FEED_ATOM = None

AUTHOR_FEED_ATOM = None

AUTHOR_FEED_RSS = None

DEFAULT_PAGINATION = 10

DEFAULT_CATEGORY = 'pages'

AUTHORS_SAVE_AS = ''

CATEGORIES_SAVE_AS = ''

ARCHIVES_SAVE_AS = ''

ARTICLE_SAVE_AS = '{category}/{slug}/index.html'

ARTICLE_URL = '{category}/{slug}/'

AUTHOR_URL = ''

AUTHOR_SAVE_AS = ''

TAG_SAVE_AS = ''

INDEX_SAVE_AS = 'pages/materials.html'

THEME_STATIC_DIR = 'style'

DELETE_OUTPUT_DIRECTORY = True

MARKUP = ['md', 'ipynb']

PLUGIN_PATHS = ['plugins']

# FOR PAVLOS COMPUTER .....
# PLUGINS = ['tipue_search']
PLUGINS = ['ipynb.markup', 'tipue_search']

# from pelican_jupyter import markup as nb_markup
# PLUGINS = ['ipynb.markup', 'tipue_search']
# PLUGINS = [nb_markup, 'ipynb.markup', 'tipue_search']

IGNORE_FILES = ['.#*', '.ipynb_checkpoints', 'README.md', "*.html", "__pycache__", "*.pdf", "*.pptx", ".placeholder",
                ".DS_Store", "*.ipynb-meta", "*.csv", "*.json", "*.txt", "*.xmls"]

STATIC_PATHS = ['lectures', 'labs', 'homeworks', 'a-sections', 'sections', 'wiki', 'images', 'projects', 'slides',
                'data']

DIRECT_TEMPLATES = ['index', 'search', 'tags', 'category']

import re

JINJA_FILTERS = {
    'original_content': lambda x: re.search(r"content/(.*)", x).group(1)
}

USE_FOLDER_AS_CATEGORY = False

import logging

LOG_FILTER = [
    (logging.WARN, "Empty alt attribute for image %s in %s"),
    (logging.WARN, "Meta tag in file %s does not have a 'name' attribute, skipping. Attributes: content=%s")
]
