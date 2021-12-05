#!/bin/bash
git add --all .
if [ "$#" -ne 1 ]; then
   git commit
else
	if [ "$1" == "f" ]; then
		git commit -m "minor fix"
	else
		git commit -m "$1"
	fi
fi
git push origin master