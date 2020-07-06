#!/usr/bin/env bash

convert color.png -set colorspace Gray gray.png
convert gray.png -gamma 0.3 gamma.png
convert gamma.png -fill black -colorize 40% darker.png
