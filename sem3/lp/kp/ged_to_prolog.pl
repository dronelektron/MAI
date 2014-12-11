#!/usr/bin/perl

use strict;
use warnings;
use utf8;

binmode(STDOUT, ":utf8");

open(my $hPid, '<:utf8', $ARGV[0]) or die("Error. Can\'t open a file\n");

my %hash;
my $str = "";
my $husb = "";
my $wife = "";

$hash{''} = 'nil';

while (my $line = <$hPid>)
{
	$line =~ tr/\r\n//d;
	
	if ($line =~ m/0 \@I([0-9]+)\@ INDI/)
	{
		$str = $1;
	}
	elsif ($line =~ m!1 NAME (.*) /.*?/!)
	{
		$hash{$str} = $1;
	}
	elsif ($line =~ m/2 (SURN|_MARNM) (.*)/)
	{
		$hash{$str} = $2 . ' ' . $hash{$str};
	}
	elsif ($line =~ m/0 \@F([0-9]+)\@ FAM/)
	{
		$husb = '';
		$wife = '';
	}
	elsif ($line =~ m/1 HUSB \@I([0-9]+)\@/)
	{
		$husb = $1;
	}
	elsif ($line =~ m/1 WIFE \@I([0-9]+)\@/)
	{
		$wife = $1;
	}
	elsif ($line =~ m/1 CHIL \@I([0-9]+)\@/)
	{
		print qq/parents("$hash{$1}", "$hash{$husb}", "$hash{$wife}").\n/;
	}
}

close($hPid);
