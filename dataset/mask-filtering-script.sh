for f in masks-f/* ; do
	echo $f;
	id="${f:8:8}";
	evince $f;
	read -ep "Enter index for ${id}: " index;

	index="$(($index-1))";

	neg=-1;

	if [ "$index" -ne "$neg" ]; then
		indexed="${f}[${index}]";
		echo $indexed;
		convert $indexed filtered/${id}.png;
	fi;
	rm $f;		
done
