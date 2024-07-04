from setuptools import setup, find_packages

setup(
    name="flash_attention_x",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        # List your package dependencies here
        'triton>=2.3.0',
    ],
    author="Xiaotian Han",
    author_email="your.email@example.com",
    description="A flash attention(s) implementation in triton.",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url="https://github.com/ahxt/flash-attention-x",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)