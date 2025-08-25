.PHONY: pdf
pdf:
	cd thesis && latexmk -pdf -interaction=nonstopmode main.tex
