<?php
	if ($_GET['all_cars'])
	{
		$xml = new DOMDocument('1.0', 'utf-8');
		$xml->load('cars.xml');

		$xsl = new DOMDocument('1.0', 'utf-8');
		$xsl->load('cars.xslt');

		$proc = new XSLTProcessor();
		$proc->importStylesheet($xsl);
		$proc->setParameter('', 'auto_body_type', $_GET['auto_body_type']);
		$result = $proc->transformToXml($xml);

		echo $result;
	}
?>
