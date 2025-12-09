import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'ros2_rag'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages() + [
        'llm_wrappers.deepseek_wrapper',
        'llm_wrappers.faiss_wrapper',
        'llm_wrappers.flan_t5_wrapper',
        'llm_wrappers.qwen_wrapper'
    ],
    package_dir={
        'llm_wrappers.deepseek_wrapper': 'llm_wrappers/deepseek_wrapper',
        'llm_wrappers.faiss_wrapper': 'llm_wrappers/faiss_wrapper',
        'llm_wrappers.flan_t5_wrapper': 'llm_wrappers/flan_t5_wrapper',
        'llm_wrappers.qwen_wrapper': 'llm_wrappers/qwen_wrapper'
    },

    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
         glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yml')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'test', 'data'),
         glob('test/data/*')),
        (os.path.join('share', package_name, 'test', 'config'),
         glob('test/config/*')),
    ],
    install_requires=[
        'setuptools',
    ],
    tests_require=['pytest'],
    zip_safe=True,
    maintainer='Aitor Ibarguren',
    maintainer_email='aitor.ibarguren.s@gmail.com',
    description='ROS2 RAG Package',
    license='Apache 2.0',
    entry_points={
        'console_scripts': [
            "ros2_rag = ros2_rag.ros2_rag:main"
        ],
    },
)
