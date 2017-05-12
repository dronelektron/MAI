#!/bin/bash

cd ~/Рабочий\ стол
mkdir -p lab2
cd lab2
cp ~/Dropbox/MAI/sem8/pgp/labs/2/makefile .
cp ~/Dropbox/MAI/sem8/pgp/labs/2/*.h .
cp ~/Dropbox/MAI/sem8/pgp/labs/2/*.cu .
cd ..
tar cf lab2.tar lab2
rm -rf lab2
