#!/bin/bash

cd ~/Рабочий\ стол
mkdir -p lab4
cd lab4
cp ~/Dropbox/MAI/sem8/pgp/labs/4/makefile .
cp ~/Dropbox/MAI/sem8/pgp/labs/4/*.h .
cp ~/Dropbox/MAI/sem8/pgp/labs/4/*.cu .
cd ..
tar cf lab4.tar lab4
rm -rf lab4
