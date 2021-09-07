echo "Running As: $(whoami):$(id -gn)"
installation_dir='/opt/miniconda'

conda info --envs
conda deactivate
conda remove --name resilience --all
#conda info --envs
conda clean --all

rm -rf ${installation_dir}
rm -rf ~/.condarc ~/.conda ~/.continuum
rm -rf ~/.cache/pip
rm glibc.apk
#rm /opt/conda/bin/conda clean -tipsy
#rm ~/miniconda.sh
