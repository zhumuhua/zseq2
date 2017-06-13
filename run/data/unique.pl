#!/usr/bin/perl


use warnings;
use strict;

my %hashset;
while(<STDIN>)
{
    chomp;
    $hashset{$_} = 1;
}

foreach my $key (keys %hashset)
{
    print "$key\n";
}
