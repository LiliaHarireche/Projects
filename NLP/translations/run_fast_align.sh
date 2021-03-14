#!/bin/bash

#aws translations alignement
StringArray1="../../gender-bias-repo/translations/aws/en-ar.txt ../../gender-bias-repo/translations/aws/en-de.txt ../../gender-bias-repo/translations/aws/en-fr.txt ../../gender-bias-repo/translations/aws/en-uk.txt ../../gender-bias-repo/translations/aws/en-ru.txt ../../gender-bias-repo/translations/aws/en-es.txt ../../gender-bias-repo/translations/aws/en-it.txt ../../gender-bias-repo/translations/aws/en-he.txt"

for var1 in $StringArray1
do
    ./fast_align -i $var1 -d -o -v > ../../_aligns/aws/forward${var1:41:5}.align  # perform fast_align and save the alignement
done

#google trtranslations alignement
StringArray2="../../gender-bias-repo/translations/google/en-ar.txt ../../gender-bias-repo/translations/google/en-de.txt ../../gender-bias-repo/translations/google/en-fr.txt ../../gender-bias-repo/translations/google/en-uk.txt ../../gender-bias-repo/translations/google/en-ru.txt ../../gender-bias-repo/translations/google/en-es.txt ../../gender-bias-repo/translations/google/en-it.txt ../../gender-bias-repo/translations/google/en-he.txt"

for var1 in $StringArray2
do
    ./fast_align -i $var1 -d -o -v > ../../_aligns/google/forward${var1:43:5}.align  # perform fast_align and save the alignement 
done

#bing translations alignement
StringArray3="../../gender-bias-repo/translations/bing/en-ar.txt ../../gender-bias-repo/translations/bing/en-de.txt ../../gender-bias-repo/translations/bing/en-fr.txt ../../gender-bias-repo/translations/bing/en-uk.txt ../../gender-bias-repo/translations/bing/en-ru.txt ../../gender-bias-repo/translations/bing/en-es.txt ../../gender-bias-repo/translations/bing/en-it.txt ../../gender-bias-repo/translations/bing/en-he.txt"

for var1 in $StringArray3
do
    ./fast_align -i $var1 -d -o -v > ../../_aligns/bing/forward${var1:41:5}.align # perform fast_align and save the alignement
done

#systran translations alignement
StringArray4="../../gender-bias-repo/translations/systran/en-ar.txt ../../gender-bias-repo/translations/systran/en-de.txt ../../gender-bias-repo/translations/systran/en-fr.txt ../../gender-bias-repo/translations/systran/en-uk.txt ../../gender-bias-repo/translations/systran/en-ru.txt ../../gender-bias-repo/translations/systran/en-es.txt ../../gender-bias-repo/translations/systran/en-it.txt ../../gender-bias-repo/translations/systran/en-he.txt"

for var1 in $StringArray4
do
    ./fast_align -i $var1 -d -o -v > ../../_aligns/systran/forward${var1:44:5}.align  # perform fast_align and save the alignement
done
