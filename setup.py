from setuptools import setup, find_packages

setup(
    name="xai-cv-matcher",
    version="1.0.0",
    description="Explainable AI CV–Job Skill Matching System",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "spacy>=3.7.0",
        "sentence-transformers>=2.7.0",
        "scikit-learn>=1.4.0",
        "numpy>=1.26.0",
        "pdfplumber>=0.11.0",
        "python-docx>=1.1.0",
        "shap>=0.45.0",
        "rich>=13.7.0",
        "click>=8.1.0",
        "pydantic>=2.6.0",
    ],
    extras_require={
        "ai": ["anthropic>=0.27.0"],
    },
    entry_points={
        "console_scripts": [
            "cv-match=cv_matcher.cli:main",
        ],
    },
)
