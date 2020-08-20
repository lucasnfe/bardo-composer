import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="bardo-composer",
    version="0.0.1",
    author="Lucas N. Ferreira",
    author_email="lferreira@ucsc.edu",
    description="Computer-Generated Music for Tabletop Role-Playing Games",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lucasnfe/bardo-composer",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
