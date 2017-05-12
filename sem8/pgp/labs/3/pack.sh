#!/bin/bash

cd ~/Рабочий\ стол
mkdir -p lab3
cd lab3
cp ~/Dropbox/MAI/sem8/pgp/labs/3/makefile .
cp ~/Dropbox/MAI/sem8/pgp/labs/3/*.h .
cp ~/Dropbox/MAI/sem8/pgp/labs/3/*.cu .
cd ..
tar cf lab3.tar lab3
rm -rf lab3
