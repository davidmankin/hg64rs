hg64rs
=============

Port of [hg64](https://github.com/fanf2/hg64) to Rust.

TODO explain more.

Testing
---------
There are some light unit tests to see how it works overall.

There is a validate() method on the original code that is used to ensure
that every generated histogram meets some sanity checks.

There are some tests that are ported from the original project that
use random input and report on performance.  These don't make
assertions.

To run the tests use `cargo test`.  To run the tests in release mode
(so the timings are more accurate) use `cargo test -r`.  And to show
the outputs that report on numeric and timing performance use
`cargo test -r -- --nocapture`.



Licence
-------

Written by David Mankin, based on code by Tony Finch.

Permission is hereby granted to use, copy, modify, and/or
distribute this software for any purpose with or without fee.

This software is provided 'as is', without warranty of any kind.
In no event shall the authors be liable for any damages arising
from the use of this software.

    SPDX-License-Identifier: 0BSD OR MIT-0

_[this is a zero-conditions libre software licence](https://dotat.at/0lib.html)_
