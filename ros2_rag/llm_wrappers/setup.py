from setuptools import find_packages, setup

setup(
    name='llm_wrappers',
    version='0.3',
    packages=find_packages(),
    description='LLM wrapper classes in Python for a simple development of LLM-based applications',
    author='Aitor Ibarguren',
    url='https://github.com/aitor-ibarguren/llm_wrappers',
    author_email='aitor.ibarguren.s@gmail.com',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
)
