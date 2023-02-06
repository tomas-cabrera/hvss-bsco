/* -*- linux-c -*- */
/* exchange_binsingle.c

   Copyright (C) 2002-2004 John M. Fregeau

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 2 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program; if not, write to the Free Software
   Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

   Note:
   =====

   Implementation of the post-Newtonian relativistic terms, as well as some
   modification of the code (parse arguments instead of hard-wired etc, look
   for "PAS" to see what I changed) by:

   Pau Amaro-Seoane (PAS)
   pau.amaro.seoane@gmail.com
   http://astro-gr.org/

   Please let me know whether you intend to use my modifications and
   contact me.

*/

/*
 * usage: see the function usage(), below.
 */

/*
 *  Mod, Jul 09, PAS.  Increase output to output ainit, bmax and vinf to the ofp
 *  file.  New post-processor "prawn" wants these things.
 *
 *  added -r switch and merged rel and nonrel branches.
 */

#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <getopt.h>

#include <err.h>
#include <fcntl.h>
#include <unistd.h>
#include <limits.h>

#include <gsl/gsl_rng.h>
#include "fewbody.h"

extern char *__progname;

/* John's original suggestion had 100.*FB_CONST_AU for default a1.
 * The .1 value is from a version of scatter_binsingle.c
 * PAS 7/08
 */


#define FB_A1 (0.1 * FB_CONST_AU)
#define FB_E1 0.0
#define FB_VINF (50.*1.e+05)
#define FB_M0_DEF 10.0
#define FB_M10_DEF 10.0
#define FB_M11_DEF 10.0

#define NEXPT 50000


void usage(void )
{
/*
 * Rectified to Unix conventions   PAS
 */
	fprintf(stderr, "usage: %s [-h] [-a axis] [-e eccentricity] ", __progname);
	fprintf(stderr, "[-v vinf] [-n experiments] [-r] [-s seed]\n");
	fprintf(stderr, "\t[--m0 val] [--m10 val] [--m11 val]\n");
	fprintf(stderr, "  -a --a1 <a1>     Semimajor axis of binary, default %.6g AU\n", FB_A1/FB_CONST_AU);
	fprintf(stderr, "  -e --e1 <e1>     Eccentricity of binary 0, default %.6g\n", FB_E1);
	fprintf(stderr, "  -v --vinf <vinf> Value of v_inf, default %.6g km/sec\n", FB_VINF/1.e+05);
	fprintf(stderr, "  -n <N>           Number of experiments, default %d\n", NEXPT);
	fprintf(stderr, "  -r		    Select \"relativistic\" option. Default, non-rel\n");
	fprintf(stderr, "  -s <seed>        Random number seed, default is random\n");
	fprintf(stderr, " --m0 <val>    Mass of interloper, default %.6g MSun\n", FB_M0_DEF);
	fprintf(stderr, " --m10 <val>   Mass of binary member 0, default %.6g MSun\n", FB_M10_DEF);
	fprintf(stderr, " --m11 <val>   Mass of binary member 1, default %.6g MSun\n", FB_M11_DEF);
	fprintf(stderr, "  -h --help        Display this text and exit\n");
	exit(1);
}

/* calculate the units used; here the unit of velocity is the critical velocity, the unit
   of length is the binary's semimajor axis, G=1, and the other units are derived */
int calc_units(fb_obj_t *obj[2], fb_units_t *units)
{
	units->v = sqrt(FB_CONST_G*(obj[0]->m + obj[1]->m)/(obj[0]->m * obj[1]->m) * \
			(obj[1]->obj[0]->m * obj[1]->obj[1]->m / obj[1]->a));
	units->l = obj[1]->a;
	units->t = units->l / units->v;
	units->m = units->l * fb_sqr(units->v) / FB_CONST_G;
	units->E = units->m * fb_sqr(units->v);

	return(0);
}

/* the main attraction */
int main(int argc, char **argv)
{
/*
 * Who, What, Where, When, Why...
 *
 * Output to:
 *	scatter_params_NNNNN.dat, where NNNNN is the pid of
 *      this process, written in a directory specified by
 *	environment variable OUTDIR, or if OUTDIR is not set,
 *      in the current working directory.
 */
	/* rectified declarations to Unix norms  PAS */

	int	i;
	int	j;

	int	expt;
	int	fd;
	int	ngood=0;	/* Move initializations? */
	int	nbad=0;
	int	ncoll=0;
	int	nexpt;
	int	bid;
	int	sid;
	int	relativistic=0;	/* Jul 09 PAS */

	unsigned long int	seed;

	double	a1;
	double	b;
	double	bmax;
	double	e1;
	double	inita1;
	double	inite1;
	double	m0;
	double	m10;
	double	m11;
	double  m0_param;
	double  m10_param;
	double  m11_param;	/* _param stuff, PAS 07/09 */
	double	m1;
	double	r0;
	double	r10;
	double	r11;
	double	rtid;
	double	t;
	double	vinf;
	double	vinf_param;

	char	*fname_root;
	char	fname[PATH_MAX];
	FILE	*ofp;

	const char *short_opts = "a:e:v:n:s:hr";
	const struct option long_opts[] = {
		{"a1", required_argument, NULL, 'a'},
		{"e1", required_argument, NULL, 'e'},
		{"m0", required_argument, NULL, 256 },
		{"m10", required_argument, NULL, 257 },
		{"m11", required_argument, NULL, 258 },
		{"vinf", required_argument, NULL, 'v'},
		{"help", no_argument, NULL, 'h'},
		{NULL, 0, NULL, 0}
	};


	/* these are fewbody data types */
	fb_hier_t hier;
	fb_input_t input;
	fb_ret_t retval;
	fb_units_t units;
	char string1[1024], string2[1024];
	/* you need the random number generator for randomizing the binary's
	 * phase and orientation */
	gsl_rng *rng;
	const gsl_rng_type *rng_type=gsl_rng_mt19937;


	/*
	 * Read a seed for the random number generator from the system
	 * source of cryptographicly semi-strong random numbers.  If strong
	 * random number seed is desired, use /dev/srandom.  The quality of
	 * Linux's /dev/srandom and /dev/urandom is not known to me.
	 *
	 * I know nothing of OSX.
	 *
	 * The gsl routine selected, mt19937, is said to be good for simulation
	 * work.
	 *
	 * The seed obtained here should be stored in the output for future
	 * reference.  Where to do that is TBD.  Perhaps in the log file.
	 *
	 * PAS (Jul 26 08)
	 */

	fd = open("/dev/urandom", O_RDONLY);
	if(fd == -1)
		err(1,"/dev/urandom");
	i = read(fd, &seed, sizeof(unsigned long int));
	if(i!= sizeof(unsigned long int))
		errx(1, "trouble reading /dev/urandom: %d", i);
	close(fd);


	/* set other parameters to default values */
	inita1 = FB_A1;
	inite1 = FB_E1;
	nexpt = NEXPT;
	vinf_param= FB_VINF;

	m0_param = FB_M0_DEF * FB_CONST_MSUN;
	m10_param = FB_M10_DEF * FB_CONST_MSUN;
	m11_param = FB_M11_DEF * FB_CONST_MSUN;

	while ((i = getopt_long(argc, argv, short_opts, long_opts, NULL)) != -1) {
		switch (i) {
		case 'a':
			inita1 = atof(optarg) * FB_CONST_AU;
			break;
		case 'e':
			inite1 = atof(optarg);
			if (inite1 >= 1.0) {
				fprintf(stderr, "e0 must be less than 1\n");
				usage();
				/* NOT REACHED */
			}
			break;
		case 'v':
			vinf_param = atof(optarg) * 1.e+05;
			break;
		case 'h':
			fb_print_version(stderr);
			fputc('\n', stderr);
			usage();
			/* NOT REACHED */
		case 'n':
			nexpt = atoi(optarg);
			break;
		case 'r':
			relativistic = 1;
			break;
		case 's':
			seed = atol(optarg);
			break;
		case 256:	/* m0 */
			m0_param = atof(optarg) * FB_CONST_MSUN;
			break;
		case 257:	/* m10 */
			m10_param = atof(optarg) * FB_CONST_MSUN;
			break;
		case 258:	/* m10 */
			m11_param = atof(optarg) * FB_CONST_MSUN;
			break;
		case ':':
			warnx("missing option argument");
			usage();
			/* NOT REACHED */
		case '?':
		default:
			warnx("unknown argument");
			usage();
			/* NOT REACHED */
		}
	}

	argc -= optind;
	argv += optind;

	/* check to make sure there was nothing extra on the command line */
	if (argc) {
		warnx("extra elements on command line, staring with %s",
			argv[0]);
		usage();
	}


	/* set parameters */
	input.ks = 0;	/* KS regularization flag; it must be off to perform physical collisions properly */
	input.tstop = 1.0e9;	/* maximum stopping time */
	input.Dflag = 0;	/* print dynamical data to stdout? */
	input.dt = 1.0;		/* approximate output interval */
	input.tcpustop = 600.0;	/* give up after this many seconds of cpu time */
	input.absacc = 1.0e-9;	/* integrator's absolute accuracy */
	input.relacc = 1.0e-9;	/* integrator's relative accuracy */
	input.ncount = 500;	/* number of integration steps between classification hierarchies */
	input.tidaltol = 1.0e-5;	/* tidal tolerance: this is what most affects energy conservation */
	/* DEBUG: set speed tolerance here */
//	input.speedtol = 5.0e-02;
	input.speedtol = 1.0e+10;
	/* DEBUG */
	input.fexp = 3.0;	/* expansion factor of merger products; 3 is a good value for main sequence stars */

	/*
	 * These being the sole difference I can find between nonrel and rel
	 * versions, they should be made command line option(s) and the
	 * nonrel and rel fork merged.  PAS 7/08
	 *                              done PAS 7/09
	 */

	if(relativistic) {
		input.PN1 = 1;
		input.PN2 = 1;
		input.PN25 = 1;
	}
	else {
		input.PN1 = 0;
		input.PN2 = 0;
		input.PN25 = 0;
	}

	/* DEBUG */
	input.PN3 = 0;
	input.PN35 = 0;

	input.firstlogentry[0] = '\0';	/* you can store log info in the output stream */
	fb_debug = 0;			/* global variable (the only one) controlling debug information */

	/* malloc hier */
	hier.nstarinit = 3;
	fb_malloc_hier(&hier);

	/* initialize GSL rng */
	gsl_rng_env_setup();
	rng = gsl_rng_alloc(rng_type);
	gsl_rng_set(rng, seed);

	/* open output file and write header */
	/* DEBUG: set file name here */

	/* all PAS in this region now Jul/08 */
	fname[0] = '\0';
	fname_root = getenv("OUTDIR");
	snprintf(fname, sizeof(fname), "%s/scatter_params_%4.4u.dat",
		fname_root?fname_root:".", getpid());
	if(!(ofp = fopen(fname, "w")))
		err(1,"unable to open %s", fname);
	if(setvbuf(ofp, NULL, _IOLBF, 0) == EOF)
		warn("linebuffering on %s failed", fname);

	/* an alternative to exiting on error is to set ofp to stdout, but
	 * this may have undesirable consequences in the long run.
	 */

	/* DEBUG */
	fprintf(ofp, "#1:v_bin[km/s] #2:v_single[km/s] #3:a[AU] #4:e #5:m1[MSUN] ");
	fprintf(ofp, "#6:m2[MSUN] #7:x1[cm] #8:y1[cm] #9:z1[cm] #10:vx1[km/s] ");
	fprintf(ofp, "#11:vy1[km/s] #12:vz1[km/s] #13:x2[cm] #14:y2[cm] #15:z2[cm] ");
	fprintf(ofp, "#16:vx2[km/s] #17:vy2[km/s] #18:vz2[km/s] #19:e_init");
        fprintf(ofp, "#20: a_init[AU] #21: bmax[AU] #22: vinf[km/s]\n");

	/* loop through experiments */
	for (i=0; i<nexpt; i++) {	/* PAS */
		expt = i;	/* shadowing i is a wart */

		m0 = m0_param;
		m10 = m10_param;
		m11 = m11_param; 	/*
					 * PAS: see use of m0 below
					 * and m1.  they are confused.
					 * see old version of code where m0,m10,m11 are
					 * mysteriously refreshed for each iteration of this
					 * loop.
					 *
					 * there is strangeness with units...
					 * when hier is loaded with m0,m10,m11 below, the
					 * m's are in "cgs" (value 10*FB_CONST_MSUN.)  When
					 * the hier[] stuff is printed out, below, they have
					 * mysteriously acquired a factor of units.m, a mysterious
					 * number.  Presumably this happens somewher in an fb_routine.
					 *
					 * Notice the strange use of m0 and a new variable, m1, as
					 * typing aids below, calculation of rtid.  Is that all they are?
					 * This was a subtle bug cooking, and restoring the m's here
					 * will, I hope, prevent its ripening into a wicked mess.
					 *  Note how I have yanked the old code out of the loop to before
					 * the getopt call.
					 *  I'm tempted to riddle the code with asserts.
					 * Jul 09
					 */

		r0 = FB_REFF_BH * FB_CONST_G * m0 / fb_sqr(FB_CONST_C);
		r10 = FB_REFF_BH * FB_CONST_G * m10 / fb_sqr(FB_CONST_C);
		r11 = FB_REFF_BH * FB_CONST_G * m11 / fb_sqr(FB_CONST_C);
		a1 = inita1;
		e1 = inite1;

		/* flatten hier */
		t = 0.0;
		hier.nstar = 3;
		fb_init_hier(&hier);

		/* create binary */
		hier.hier[hier.hi[2]+0].obj[0] = &(hier.hier[hier.hi[1]+1]);
		hier.hier[hier.hi[2]+0].obj[1] = &(hier.hier[hier.hi[1]+2]);
		hier.hier[hier.hi[2]+0].t = t;

		/* give the objects some properties */
		for (j=0; j<hier.nstar; j++) {
			hier.hier[hier.hi[1]+j].ncoll = 1;
			hier.hier[hier.hi[1]+j].id[0] = j;
			snprintf(hier.hier[hier.hi[1]+j].idstring, FB_MAX_STRING_LENGTH, "%d", j);
			hier.hier[hier.hi[1]+j].n = 1;
			hier.hier[hier.hi[1]+j].obj[0] = NULL;
			hier.hier[hier.hi[1]+j].obj[1] = NULL;
			hier.hier[hier.hi[1]+j].Eint = 0.0;
			hier.hier[hier.hi[1]+j].Lint[0] = 0.0;
			hier.hier[hier.hi[1]+j].Lint[1] = 0.0;
			hier.hier[hier.hi[1]+j].Lint[2] = 0.0;
		}

		/* set radii */
		hier.hier[hier.hi[1]+0].R = r0;
		hier.hier[hier.hi[1]+1].R = r10;
		hier.hier[hier.hi[1]+2].R = r11;

		/* masses */
		hier.hier[hier.hi[1]+0].m = m0;
		hier.hier[hier.hi[1]+1].m = m10;
		hier.hier[hier.hi[1]+2].m = m11;

		hier.hier[hier.hi[2]+0].m = m10 + m11;

		/* orbital parameters */
		hier.hier[hier.hi[2]+0].a = a1;
		hier.hier[hier.hi[2]+0].e = e1;

		/* make obj's point to the correct nodes in the hier */
		hier.obj[0] = &(hier.hier[hier.hi[1]+0]);
		hier.obj[1] = &(hier.hier[hier.hi[2]+0]);
		hier.obj[2] = NULL;

		/* get the units and normalize */
		calc_units(hier.obj, &units);
		fb_normalize(&hier, units);	/* presumably this is where the unit magic is done */

		/* DEBUG: set v_infinity and maximum impact parameter here */
		/* set v_inf to ~20 km/s, a reasonable value for a BH core */
		/* vinf = 20.0e5 / units.v; */
		/* set v_inf to something less than 1 to get a resonant encounter */
		/*vinf = 0.2; commented out Pau (02/07/2009)  */
                /* vinf = 50.0e5/units.v; / *10.0e5 is 10km/s in cgs, and units.v is the unit of velocity in cgs Pau (02/07/2009) */
		/*
		 * Jul 09.  Vinf is now a command line flag, default value FB_VINF
		 */


		vinf = vinf_param / units.v;


		/* maximum impact parameter, from Hut & Bahcall (1983) (x2 to make sure we catch everything interesting) */
		/* bmax is fraught with magic numbers */
		bmax = 1.0 * (4.0/vinf + 0.6);
		/* DEBUG */
		/* sample impact parameter uniformly in area */
		b = sqrt(gsl_rng_uniform(rng)) * bmax;

		/* analytically move objects along hyperbolic path */
		/*
		 * PAS: there appears to be some confusion here.
		 * It concerns the variables m0, m1, m10, m11.
		 * Which is which?  In the current implementation they
		 * don't seem to change; but it does look like m0 here
		 * is not the m0 used above to set masses into hier.obj[*]-> m a few lines above.
		 * By "not the same", I mean not conceptualy the same.
		 * In fact, it looks like m0,m1 are being used as a typing aide for
		 * the calc. of rtid.  But "conceptually" m0 is part of the m0, m10, m11
		 * triple, "conceptually" constant parameters.
		 */
		m0 = hier.obj[0]->m;
		m1 = hier.obj[1]->m;
		a1 = hier.obj[1]->a;
		e1 = hier.obj[1]->e;


		/* rtid is the radius at which the binary's tidal perturbation (F_tid,max/F_rel,min) is equal
		   to the tidal tolerance; we want to start the integration at this radius */
		rtid = pow(2.0*(m0+m1)/(m1*input.tidaltol), 1.0/3.0) * a1 * (1.0+e1);
		fb_init_scattering(hier.obj, vinf, b, rtid);

		/* randomly orient binary */
		fb_randorient(&(hier.hier[hier.hi[2]+0]), rng);	/* randomly orient binary */
		/* use parent node's properties to set properties of child nodes*/
		fb_downsync(&(hier.hier[hier.hi[2]+0]), t);
		/* determine parent node's properties from child nodes
		   (this step is not necessary, and was included only as a test) */
		fb_upsync(&(hier.hier[hier.hi[2]+0]), t);

		/* call fewbody! */
		retval = fewbody(input, units, &hier, &t, rng);

		/* print to screen */
		fprintf(stdout, "expt=%d retval=%d tcpu=%g DeltaEfrac=%g DeltaLfrac=%g vinf=%g bmax=%g b=%g %s (%s)\n",
			expt, retval.retval, retval.tcpu, retval.DeltaEfrac, retval.DeltaLfrac, vinf, bmax, b,
			fb_sprint_hier(hier, string1), fb_sprint_hier_hr(hier, string2));

		/* print to special data file if we have a useable exchange, i.e., the return value is 1, so the
		   calculation finished, and energy and angular momentum are conserved reasonably */

		if (retval.retval == 1) {
			ngood++;
			/* no collisions */
			if (hier.nstar == 3) {
				/* exchange or preservation */
				if (hier.nobj == 2) {
					/* test to see which object is the binary */
					if (fb_n_hier(hier.obj[0]) == 2) {
						bid = 0;
						sid = 1;
					} else {
						sid = 0;
						bid = 1;
					}
					fprintf(ofp, "%g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g\n", /* Jul09 */
						fb_mod(hier.obj[bid]->v) * units.v / 1.0e5,
						fb_mod(hier.obj[sid]->v) * units.v / 1.0e5,
						hier.obj[bid]->a * units.l / FB_CONST_AU,
						hier.obj[bid]->e,
						hier.obj[bid]->obj[0]->m * units.m / FB_CONST_MSUN,
						hier.obj[bid]->obj[1]->m * units.m / FB_CONST_MSUN,
						(hier.obj[bid]->obj[0]->x[0] - hier.obj[bid]->x[0]) * units.l,
						(hier.obj[bid]->obj[0]->x[1] - hier.obj[bid]->x[1]) * units.l,
						(hier.obj[bid]->obj[0]->x[2] - hier.obj[bid]->x[2]) * units.l,
						(hier.obj[bid]->obj[0]->v[0] - hier.obj[bid]->v[0]) * units.v / 1.0e5,
						(hier.obj[bid]->obj[0]->v[1] - hier.obj[bid]->v[1]) * units.v / 1.0e5,
						(hier.obj[bid]->obj[0]->v[2] - hier.obj[bid]->v[2]) * units.v / 1.0e5,
						(hier.obj[bid]->obj[1]->x[0] - hier.obj[bid]->x[0]) * units.l,
						(hier.obj[bid]->obj[1]->x[1] - hier.obj[bid]->x[1]) * units.l,
						(hier.obj[bid]->obj[1]->x[2] - hier.obj[bid]->x[2]) * units.l,
						(hier.obj[bid]->obj[1]->v[0] - hier.obj[bid]->v[0]) * units.v / 1.0e5,
						(hier.obj[bid]->obj[1]->v[1] - hier.obj[bid]->v[1]) * units.v / 1.0e5,
						(hier.obj[bid]->obj[1]->v[2] - hier.obj[bid]->v[2]) * units.v / 1.0e5,
						inite1,
                                                inita1 / FB_CONST_AU,                                /* Jul09 */
                                                bmax * units.l / FB_CONST_AU,                                /* Jul09 */
                                                vinf * units.v /1.e05 );                                     /* Jul09 */
					/* fflush(ofp);	  Removed.  ofp is set for linebuffering,
					 * which accomplishes the same thing transparently.
					 */
				}
				else {	/* PAS */
					fprintf(ofp, "# experiment %u; hier.nobj = %u (!=2), no scatter print out\n",
						expt, hier.nobj);
				}
			} else { /* hier.nstar != 3, so there must have been a physical collision */
				/* This rare event should be noted in the output somehow */
				ncoll++;
				fprintf(ofp, "# experiment %u had a collision\n", expt);
			}
		} else { /* calculation didn't finish, or energy or ang mom wasn't conserved */
			/* Isn't this a catastrophe?  One might dump data here. */
			nbad++;
			fprintf(ofp, "# experiment %u, retval = %d (!= 1)\n",
				expt, retval.retval);
		}
		/* incrementally update cross sections on screen */
		fprintf(stdout, "n=%d ngood=%d nbad=%d ncoll=%d\n",
			ngood+nbad, ngood, nbad, ncoll);
		fprintf(stdout, "sigma_coll=%g+/-%g\n",
			fb_sqr(bmax)*((double) ncoll)/((double) ngood) * fb_sqr(vinf),
			fb_sqr(bmax)*sqrt((double) ncoll)/((double) ngood) * fb_sqr(vinf));

	} /* loop through experiments */


	/* free stuff */
	gsl_rng_free(rng);
	fb_free_hier(hier);

	return 0;
}
