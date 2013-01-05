#
# Makefile for hammer.
#
# Kay <kay21s AT gmail DOT com>

all: static 
static:
	cd src ; $(MAKE) $(AM_MAKEFLAGS) $@
	cd libgpucrypto; $(MAKE) $(AM_MAKEFLAGS) $@
	cd openssl-1.0.1c; $(MAKE) $(AM_MAKEFLAGS) $@
clean:
	cd src ; $(MAKE) $(AM_MAKEFLAGS) $@
	cd libgpucrypto; $(MAKE) $(AM_MAKEFLAGS) $@
	cd openssl-1.0.1c; $(MAKE) $(AM_MAKEFLAGS) $@

#EOF
