<?xml version="1.0"?>
<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
<xsl:output encoding="utf-8" indent="yes" method="html"/>
<xsl:param name="auto_body_type"></xsl:param>
	<xsl:template match="/cars">
		<xsl:text disable-output-escaping="yes">&lt;!DOCTYPE html&gt;&#10;</xsl:text>
		<html>
			<head>
				<title>Характеристики автомобилей</title>
			</head>
			<body>
				<table border="1">
					<tr>
						<th>Наименование</th>
						<th>Модификация (двигатель)</th>
						<th>Количество дверей</th>
						<th>Мощность (л.с.)</th>
						<th>Тип кузова</th>
						<th>Количество мест</th>
					</tr>
					<xsl:for-each select="info">
						<xsl:if test="contains(@body_type, $auto_body_type)">
							<tr>
								<td>
									<xsl:value-of select="@name"/>
								</td>
								<td>
									<xsl:value-of select="@engine"/>
								</td>
								<td>
									<xsl:value-of select="@doors_count"/>
								</td>
								<td>
									<xsl:value-of select="@hp"/>
								</td>
								<td>
									<xsl:value-of select="@body_type"/>
								</td>
								<td>
									<xsl:value-of select="@slots"/>
								</td>
							</tr>
						</xsl:if>
					</xsl:for-each>
				</table>
			</body>
		</html>
	</xsl:template>
</xsl:stylesheet>
