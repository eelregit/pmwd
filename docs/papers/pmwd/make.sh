#!/usr/bin/env bash

# adapted from https://github.com/dfm/emcee/blob/main/joss/make_latex.sh
# and https://github.com/openjournals/whedon/blob/master/resources/docker-entrypoint.sh

#echo Downloading...
#rm -rf latex.template apa.csl joss-logo.png aas-logo.png
#url=https://raw.githubusercontent.com/openjournals/whedon/ca417d169c382b02ee5dc858e6651aa080c5c40a/resources/joss
## use chicago style if w/o apa.csl
#wget -q $url/{latex.template,aas-logo.png}
#wget -q $url/logo.png -O joss-logo.png
#echo Done

paper=pmwd

pandoc $paper.md -Cso $paper.tex --template latex.template
pandoc $paper.md -Co $paper.pdf --template latex.template
