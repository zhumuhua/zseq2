#!/usr/bin/perl


use warnings;
use strict;


my $index = 0;
while(<STDIN>)
{
    $index += 1;
    if($index % 2 == 1)
    {
        print "$_";
    }
}
