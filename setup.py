from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="mesh_processing_tools",
    version="0.1.0",
    author="jeevanpillay",
    author_email="jp@jeevanpillay.com",
    description="A package for 3D mesh processing and analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jeevanpillay/mesh_processing",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy",
        "trimesh",
        "pyvista",
        "pymeshlab",
        "rtree",
        "mesh-to-sdf",
    ],
)
