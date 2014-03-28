SELECT `group` AS 'Группа' FROM
(
	SELECT `group`, MAX(`informat` + `linal` + `diskr`) - MIN(`informat` + `linal` + `diskr`) AS diff
	FROM persons
	GROUP BY `group`
)
AS tab
WHERE diff = (
	SELECT MAX(diff2) FROM
	(
		SELECT MAX(`informat` + `linal` + `diskr`) - MIN(`informat` + `linal` + `diskr`) AS diff2
		FROM persons
		GROUP BY `group`
	)
	AS tab2
	LIMIT 1
)
