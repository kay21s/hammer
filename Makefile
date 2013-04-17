#
# Makefile for hammer.
#
# Kay <kay21s AT gmail DOT com>

all: 
	cd libgpucrypto; $(MAKE) $(AM_MAKEFLAGS) $@
	cd src ; $(MAKE) $(AM_MAKEFLAGS) $@
clean:
	cd libgpucrypto; $(MAKE) $(AM_MAKEFLAGS) $@
	cd src ; $(MAKE) $(AM_MAKEFLAGS) $@

#EOF
