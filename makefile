SHELL := /usr/bin/env bash

# configure engine
python := python
pip := pip
CC = icc

# Main Targets ########################################################################################################################################################################################

cython:
	CC=$(CC) python setup.py build_ext --inplace

test: pytest pep8
	coverage html
testFull: pytest pep8 pylint
	coverage html

clean:
	rm -f .coverage
	rm -rf htmlcov tail.egg-info build
	find . -type f -name "*.py[co]" -delete -or -type d -name "__pycache__" -delete -or -type f -name "*.c" -delete -or -type f -name "*.so" -delete

# Making dependancies #################################################################################################################################################################################

# maintenance #########################################################################################################################################################################################

# Deploy to PyPI
## by Travis, properly git tagged
pypi:
	git tag -a v$$($(python) setup.py --version) -m 'Deploy to PyPI' && git push origin v$$($(python) setup.py --version)
## Manually
pypiManual:
	$(python) setup.py register -r pypitest && $(python) setup.py sdist upload -r pypitest && $(python) setup.py register -r pypi && $(python) setup.py sdist upload -r pypi

init:
	$(pip) install -r requirements.txt
	$(pip) install -r tests/requirements.txt

dev:
	$(pip) install -e .[test]

pytest:
	$(python) -m pytest -vv --cov=tail tests

# check python styles
pep8:
	pep8 . --ignore=E402,E501,E731
pep8Strict:
	pep8 .
pyflakes:
	pyflakes .
flake8:
	flake8 .
pylint:
	pylint tail

# cleanup python
autopep8:
	autopep8 . --recursive --in-place --pep8-passes 2000 --verbose
autopep8Aggressive:
	autopep8 . --recursive --in-place --pep8-passes 2000 --verbose --aggressive --aggressive

# pasteurize
past:
	pasteurize -wnj 4 .
