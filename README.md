<p align="center">
<a href="https://github.com/showyourwork/showyourwork">
<img width = "450" src="https://raw.githubusercontent.com/showyourwork/.github/main/images/showyourwork.png" alt="showyourwork"/>
</a>
<br>
<br>
<a href="https://github.com/tomas-cabrera/hvss-bsco/actions/workflows/build.yml">
<img src="https://github.com/tomas-cabrera/hvss-bsco/actions/workflows/build.yml/badge.svg?branch=main" alt="Article status"/>
</a>
<a href="https://github.com/tomas-cabrera/hvss-bsco/raw/main-pdf/arxiv.tar.gz">
<img src="https://img.shields.io/badge/article-tarball-blue.svg?style=flat" alt="Article tarball"/>
</a>
<a href="https://github.com/tomas-cabrera/hvss-bsco/raw/main-pdf/ms.pdf">
<img src="https://img.shields.io/badge/article-pdf-blue.svg?style=flat" alt="Read the article"/>
</a>
</p>

# Runaway and Hypervelocity Stars from Compact Object Encounters in Globular Clusters

The [showyourwork!](https://show-your.work/en/latest/) repository for this paper, the latter of which investigates how binary-single encounters in the Milky Way population of globular clusters produces runaway and hypervelocity stars.

showyourwork! was added after much of the simulation for this project had been done, and so documentation on these steps is not well-integrated with the rest.
The relevant code can be found in [src/previous_steps](https://github.com/tomas-cabrera/hvss-bsco/tree/main/src/previous_steps), and consists of:
- [fewbody-pn](https://github.com/tomas-cabrera/hvss-bsco/tree/main/src/previous_steps/fewbody-pn): The scattering code used for this project.  A copy of the [fewbody](https://gitlab.com/fregeau/fewbody) package described in [Fregeau+04](https://ui.adsabs.harvard.edu/abs/2004MNRAS.352....1F), with some physics modifications for black holes and other compact objects.  Almost all modifications have the string "HVSS" in their comment.
- [scripts_original](https://github.com/tomas-cabrera/hvss-bsco/tree/main/src/previous_steps/scripts_original): The scripts used to generate the binary-single encounter products from <code>CMC</code> output; the important files have "cmc" in their names.  Some groups of files are as follows:
    - **binsingle_extractor.ipynb**: the Jupyter Notebook used to read data from the <code>CMC</code> output files.  Requires <code>cmc_parser</code> from [CMC-Utilities](https://github.com/ClusterMonteCarlo/CMC-Utilities).
    - **realize**: The code used to call Fewbody on the <code>CMC</code> encounter population.
    - **extract**(deprecated): Was used to turn encounter-delineated data into ejecta-deliniated data.  Not used for final dataset.
    - **hbin**(deprecated): Was used to bin data for faster histogram generation.  Not used for final figures.

The data products may be found on [Zenodo](https://zenodo.org/record/7599871), including ejecta populations delineated by either <code>CMC</code> model or Milky Way globular cluster.  The <code>data.tar.gz</code> file can be unzipped into the <code>data</code> folder in this repository, after which the document should be able to be generated. 
