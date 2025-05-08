from setuptools import setup, find_packages

setup(
    name="email-wizard-assistant",
    version="0.1.0",
    description="An AI-powered Email Wizard Assistant using RAG technology",
    author="Hamza Tariq",
    packages=find_packages(),
    install_requires=[
        "fastapi",
        "uvicorn",
        "pydantic",
        "sentence-transformers",
        "chromadb",
        "numpy",
        "pandas",
        "transformers",
        "torch",
        "scikit-learn",
        "pytest",
        "jupyter",
        "matplotlib",
        "tqdm",
        "python-dotenv",
    ],
    python_requires=">=3.8",
)