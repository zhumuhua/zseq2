#!/usr/bin/perl


use warnings;
use strict;

my %hashset;
while(<STDIN>)
{
    chomp;
    s/^topic:\d+ //;
    $hashset{$_} += 1;
}

foreach my $key (keys %hashset)
{
    print "$key $hashset{$key}\n";
}
