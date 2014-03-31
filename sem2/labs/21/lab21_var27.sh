#!/usr/bin/env bash

DIR="." # Директория по-умолчанию

if [ $# -eq 1 ]; then
	DIR=$1
fi

echo "Директория: $DIR"

TEMP="$(mktemp)"
TEMP2="$(mktemp)"

for f in $(find "$DIR" -type f -name "*.*" | sed "s/ //");
do
	echo ${f##*.} >> $TEMP
done

sort $TEMP | uniq -u > $TEMP2

SFXCOUNT=$(cat $TEMP | sort -u | wc -l)

echo "Количество различных суффиксов: $SFXCOUNT"
echo "Список уникальных суффиксов:"

cat "$TEMP2"

rm "$TEMP"
rm "$TEMP2"
