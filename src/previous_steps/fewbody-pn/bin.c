/* -*- linux-c -*- */
/* bin.c

   Copyright (C) 2006 John M. Fregeau
   
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
*/

#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <getopt.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_rng.h>
#include "fewbody.h"
#include "bin.h"

/* print the usage */
void print_usage(FILE *stream)
{
	fprintf(stream, "USAGE:\n");
	fprintf(stream, "  bin [options...]\n");
	fprintf(stream, "\n");
	fprintf(stream, "OPTIONS:\n");
	fprintf(stream, "  -m --m00 <m00/MSUN>          : set mass of star 0 of binary 0 [%.6g]\n", FB_M00/FB_CONST_MSUN);
	fprintf(stream, "  -n --m01 <m01/MSUN>          : set mass of star 1 of binary 0 [%.6g]\n", FB_M01/FB_CONST_MSUN);
	fprintf(stream, "  -b --chi00 <chi_0>           : set spin of BH 0 of binary 0 (negative ignores collisions) [%.6g]\n", FB_CHI00); 
	fprintf(stream, "  -B --chi01 <chi_1>           : set spin of BH 1 of binary 0 (negative ignores collisions) [%.6g]\n", FB_CHI01);
	fprintf(stream, "  -r --r00 <r00/RSUN>          : set radius of star 0 of binary 0 [%.6g]\n", FB_R00/FB_CONST_RSUN);
	fprintf(stream, "  -g --r01 <r01/RSUN>          : set radius of star 1 of binary 0 [%.6g]\n", FB_R01/FB_CONST_RSUN);
	fprintf(stream, "  -a --a0 <a0/AU>              : set semimajor axis of binary 0 [%.6g]\n", FB_A0/FB_CONST_AU);
	fprintf(stream, "  -e --e0 <e0>                 : set eccentricity of binary 0 [%.6g]\n", FB_E0);
	fprintf(stream, "  -t --tstop <tstop/t_dyn>     : set stopping time [%.6g]\n", FB_TSTOP);
	fprintf(stream, "  -D --dt <dt/t_dyn>           : set approximate output dt [%.6g]\n", FB_DT);
	fprintf(stream, "  -c --tcpustop <tcpustop/sec> : set cpu stopping time [%.6g]\n", FB_TCPUSTOP);
	fprintf(stream, "  -A --absacc <absacc>         : set integrator's absolute accuracy [%.6g]\n", FB_ABSACC);
	fprintf(stream, "  -R --relacc <relacc>         : set integrator's relative accuracy [%.6g]\n", FB_RELACC);
	fprintf(stream, "  -N --ncount <ncount>         : set number of integration steps between calls\n");
	fprintf(stream, "                                 to fb_classify() [%d]\n", FB_NCOUNT);
	fprintf(stream, "  -z --tidaltol <tidaltol>     : set tidal tolerance [%.6g]\n", FB_TIDALTOL);
	fprintf(stream, "  -y --speedtol <speedtol>     : set speed tolerance [%.6g]\n", FB_SPEEDTOL);
	fprintf(stream, "  -x --fexp <f_exp>            : set expansion factor of merger product [%.6g]\n", FB_FEXP);
	fprintf(stream, "  -P --PN1 <PN1>               : PN1 terms on? [%d]\n", FB_PN1);
	fprintf(stream, "  -Q --PN2 <PN2>               : PN2 terms on? [%d]\n", FB_PN2);
	fprintf(stream, "  -S --PN25 <PN25>             : PN2.5 terms on? [%d]\n", FB_PN25);
	fprintf(stream, "  -T --PN3 <PN3>               : PN3 terms on? [%d]\n", FB_PN3);
	fprintf(stream, "  -U --PN35 <PN35>             : PN3.5 terms on? [%d]\n", FB_PN35);
	fprintf(stream, "  -k --ks                      : turn K-S regularization on or off [%d]\n", FB_KS);
	fprintf(stream, "  -s --seed                    : set random seed [%ld]\n", FB_SEED);
	fprintf(stream, "  -d --debug                   : turn on debugging\n");
	fprintf(stream, "  -V --version                 : print version info\n");
	fprintf(stream, "  -h --help                    : display this help text\n");
}

/* calculate the units used */
int calc_units(fb_obj_t *obj[2], fb_units_t *units)
{
	units->v = sqrt(FB_CONST_G * (obj[0]->m) / obj[0]->a);
	units->l = obj[0]->a;
	units->t = units->l / units->v;
	units->m = units->l * fb_sqr(units->v) / FB_CONST_G;
	units->E = units->m * fb_sqr(units->v);

	return(0);
}

/* the main attraction */
int main(int argc, char *argv[])
{
	int i, j;
	unsigned long int seed;
	double m00, m01, r00, r01, a0, e0;
	double chi00, chi01;
	double Ei, Lint[3], Li[3], t;
	fb_hier_t hier;
	fb_input_t input;
	fb_ret_t retval;
	fb_units_t units;
	char string1[FB_MAX_STRING_LENGTH], string2[FB_MAX_STRING_LENGTH];
	gsl_rng *rng;
	const gsl_rng_type *rng_type=gsl_rng_mt19937;
	const char *short_opts = "m:n:b:B:r:g:a:e:t:D:c:A:R:N:z:y:x:P:Q:S:T:U:k:s:dVh";
	const struct option long_opts[] = {
		{"m00", required_argument, NULL, 'm'},
		{"m01", required_argument, NULL, 'n'},
		{"chi00", required_argument, NULL, 'b'},
		{"chi01", required_argument, NULL, 'B'},
		{"r00", required_argument, NULL, 'r'},
		{"r01", required_argument, NULL, 'g'},
		{"a0", required_argument, NULL, 'a'},
		{"e0", required_argument, NULL, 'e'},
		{"tstop", required_argument, NULL, 't'},
		{"dt", required_argument, NULL, 'D'},
		{"tcpustop", required_argument, NULL, 'c'},
		{"absacc", required_argument, NULL, 'A'},
		{"relacc", required_argument, NULL, 'R'},
		{"ncount", required_argument, NULL, 'N'},
		{"tidaltol", required_argument, NULL, 'z'},
		{"speedtol", required_argument, NULL, 'y'},
		{"fexp", required_argument, NULL, 'x'},
		{"PN1", required_argument, NULL, 'P'},
		{"PN2", required_argument, NULL, 'Q'},
		{"PN25", required_argument, NULL, 'S'},
		{"PN3", required_argument, NULL, 'T'},
		{"PN35", required_argument, NULL, 'U'},
		{"ks", required_argument, NULL, 'k'},
		{"seed", required_argument, NULL, 's'},
		{"debug", no_argument, NULL, 'd'},
		{"version", no_argument, NULL, 'V'},
		{"help", no_argument, NULL, 'h'},
		{NULL, 0, NULL, 0}
	};

	/* set parameters to default values */
	m00 = FB_M00;
	m01 = FB_M01;
	chi00 = FB_CHI00;
	chi01 = FB_CHI01;
	r00 = FB_R00;
	r01 = FB_R01;
	a0 = FB_A0;
	e0 = FB_E0;
	input.ks = FB_KS;
	input.tstop = FB_TSTOP;
	input.Dflag = 0;
	input.dt = FB_DT;
	input.tcpustop = FB_TCPUSTOP;
	input.absacc = FB_ABSACC;
	input.relacc = FB_RELACC;
	input.ncount = FB_NCOUNT;
	input.tidaltol = FB_TIDALTOL;
	input.speedtol = FB_SPEEDTOL;
	input.fexp = FB_FEXP;
	input.PN1 = FB_PN1;
	input.PN2 = FB_PN2;
	input.PN25 = FB_PN25;
	input.PN3 = FB_PN3;
	input.PN35 = FB_PN35;
	seed = FB_SEED;
	fb_debug = FB_DEBUG;
	
	while ((i = getopt_long(argc, argv, short_opts, long_opts, NULL)) != -1) {
		switch (i) {
		case 'm':
			m00 = atof(optarg) * FB_CONST_MSUN;
			break;
		case 'n':
			m01 = atof(optarg) * FB_CONST_MSUN;
			break;
		case 'b':
			chi00 = atof(optarg);
			break;
		case 'B':
			chi01 = atof(optarg);
			break;
		case 'r':
			r00 = atof(optarg) * FB_CONST_RSUN;
			break;
		case 'g':
			r01 = atof(optarg) * FB_CONST_RSUN;
			break;
		case 'a':
			a0 = atof(optarg) * FB_CONST_AU;
			break;
		case 'e':
			e0 = atof(optarg);
			if (e0 >= 1.0) {
				fprintf(stderr, "e0 must be less than 1\n");
				return(1);
			}
			break;
		case 't':
			input.tstop = atof(optarg);
			break;
		case 'D':
			input.Dflag = 1;
			input.dt = atof(optarg);
			break;
		case 'c':
			input.tcpustop = atof(optarg);
			break;
		case 'A':
			input.absacc = atof(optarg);
			break;
		case 'R':
			input.relacc = atof(optarg);
			break;
		case 'N':
			input.ncount = atoi(optarg);
			break;
		case 'z':
			input.tidaltol = atof(optarg);
			break;
		case 'y':
			input.speedtol = atof(optarg);
			break;
		case 'x':
			input.fexp = atof(optarg);
			break;
		case 'P':
			input.PN1 = atoi(optarg);
			break;
		case 'Q':
			input.PN2 = atoi(optarg);
			break;
		case 'S':
			input.PN25 = atoi(optarg);
			break;
		case 'T':
			input.PN3 = atoi(optarg);
			break;
		case 'U':
			input.PN35 = atoi(optarg);
			break;
		case 'k':
			input.ks = atoi(optarg);
			break;
		case 's':
			seed = atol(optarg);
			break;
		case 'd':
			fb_debug = 1;
			break;
		case 'V':
			fb_print_version(stdout);
			return(0);
		case 'h':
			fb_print_version(stdout);
			fprintf(stdout, "\n");
			print_usage(stdout);
			return(0);
		default:
			break;
		}
	}
	
	/* check to make sure there was nothing crazy on the command line */
	if (optind < argc) {
		print_usage(stdout);
		return(1);
	}
	
	/* initialize a few things for integrator */
	t = 0.0;
	hier.nstarinit = 2;
	hier.nstar = 2;
	fb_malloc_hier(&hier);
	fb_init_hier(&hier);

	/* put stuff in log entry */
	snprintf(input.firstlogentry, FB_MAX_LOGENTRY_LENGTH, "  command line:");
	for (i=0; i<argc; i++) {
		snprintf(&(input.firstlogentry[strlen(input.firstlogentry)]), 
			 FB_MAX_LOGENTRY_LENGTH-strlen(input.firstlogentry), " %s", argv[i]);
	}
	snprintf(&(input.firstlogentry[strlen(input.firstlogentry)]), 
		 FB_MAX_LOGENTRY_LENGTH-strlen(input.firstlogentry), "\n");
	
	/* print out values of paramaters */
	fprintf(stderr, "PARAMETERS:\n");
	fprintf(stderr, "  ks=%d  seed=%ld\n", input.ks, seed);
	fprintf(stderr, "  a0=%.6g AU  e0=%.6g  m00=%.6g MSUN  m01=%.6g MSUN  r00=%.6g RSUN  r01=%.6g RSUN\n", \
		a0/FB_CONST_AU, e0, m00/FB_CONST_MSUN, m01/FB_CONST_MSUN, r00/FB_CONST_RSUN, r01/FB_CONST_RSUN);
	fprintf(stderr, "  tstop=%.6g  tcpustop=%.6g\n", \
		input.tstop, input.tcpustop);
	fprintf(stderr, "  tidaltol=%.6g  speedtol=%.6g  abs_acc=%.6g  rel_acc=%.6g  ncount=%d  fexp=%.6g\n", \
		input.tidaltol, input.speedtol, input.absacc, input.relacc, input.ncount, input.fexp);
	fprintf(stderr, "  PN1=%d  PN2=%d  PN25=%d  PN3=%d  PN35=%d\n\n", \
		input.PN1, input.PN2, input.PN25, input.PN3, input.PN35);

	/* initialize GSL rng */
	gsl_rng_env_setup();
	rng = gsl_rng_alloc(rng_type);
	gsl_rng_set(rng, seed);

	/* create binary */
	hier.hier[hier.hi[2]+0].obj[0] = &(hier.hier[hier.hi[1]+0]);
	hier.hier[hier.hi[2]+0].obj[1] = &(hier.hier[hier.hi[1]+1]);
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

	hier.hier[hier.hi[1]+0].R = 2*FB_REFF_BH*FB_CONST_G*m00/(FB_CONST_C*FB_CONST_C);
	hier.hier[hier.hi[1]+1].R = 2*FB_REFF_BH*FB_CONST_G*m01/(FB_CONST_C*FB_CONST_C);

	hier.hier[hier.hi[1]+0].m = m00;
	hier.hier[hier.hi[1]+1].m = m01;

	hier.hier[hier.hi[1]+0].chi = chi00;
	hier.hier[hier.hi[1]+1].chi = chi01;

	hier.hier[hier.hi[2]+0].m = m00 + m01;

	hier.hier[hier.hi[2]+0].a = a0;
	
	hier.hier[hier.hi[2]+0].e = e0;

	hier.obj[0] = &(hier.hier[hier.hi[2]+0]);
	hier.obj[1] = NULL;

	/* get the units and normalize */
	calc_units(hier.obj, &units);
	fb_normalize(&hier, units);

	fprintf(stderr,"lol %g %g %g\n",
	hier.hier[hier.hi[1]+0].m,
	hier.hier[hier.hi[1]+1].m
	);
	
	fprintf(stderr, "UNITS:\n");
	fprintf(stderr, "  v=%.6g km/s  l=%.6g AU  t=t_dyn=%.6g yr\n", \
		units.v/1.0e5, units.l/FB_CONST_AU, units.t/FB_CONST_YR);
	fprintf(stderr, "  M=%.6g M_sun  E=%.6g erg\n\n", units.m/FB_CONST_MSUN, units.E);

	/* place binary's center of mass at origin */
	hier.obj[0]->x[0] = 0.0;
	hier.obj[0]->x[1] = 0.0;
	hier.obj[0]->x[2] = 0.0;
	
	hier.obj[0]->v[0] = 0.0;
	hier.obj[0]->v[1] = 0.0;
	hier.obj[0]->v[2] = 0.0;

	/* orient binary so that its angular momentum points along the z-axis, its
	   Runge-Lenz vector along the x-axis, and set the mean anomaly to zero */
	hier.obj[0]->Lhat[0] = 0.0;
	hier.obj[0]->Lhat[1] = 0.0;
	hier.obj[0]->Lhat[2] = 1.0;
	
	hier.obj[0]->Ahat[0] = 1.0;
	hier.obj[0]->Ahat[1] = 0.0;
	hier.obj[0]->Ahat[2] = 0.0;
	
	hier.obj[0]->mean_anom = 0.0;

	/* trickle down the binary properties, then back up */
	for (j=0; j<1; j++) {
		fb_dprintf("obj[%d]->a=%e\n", j, hier.obj[j]->a);
		fb_dprintf("obj[%d]->e=%e\n", j, hier.obj[j]->e);
		fb_dprintf("obj[%d]->m=%e\n", j, hier.obj[j]->m);
		fprintf(stderr,"obj[%d]->a=%e\n", j, hier.obj[j]->a);
		fprintf(stderr,"obj[%d]->e=%e\n", j, hier.obj[j]->e);
		fprintf(stderr,"obj[%d]->m=%e\n", j, hier.obj[j]->m);
		fb_randorient(&(hier.hier[hier.hi[2]+0]), rng);
		fb_downsync(&(hier.hier[hier.hi[2]+j]), t);
		fb_upsync(&(hier.hier[hier.hi[2]+j]), t);
		fprintf(stderr,"obj[%d]->a=%e\n", j, hier.obj[j]->a);
		fprintf(stderr,"obj[%d]->e=%e\n", j, hier.obj[j]->e);
		fprintf(stderr,"obj[%d]->m=%e\n", j, hier.obj[j]->m);
	}
	
	/* store the initial energy and angular momentum*/
	Ei = fb_petot(&(hier.hier[hier.hi[1]]), hier.nstar) + fb_ketot(&(hier.hier[hier.hi[1]]), hier.nstar) +
		fb_einttot(&(hier.hier[hier.hi[1]]), hier.nstar);
	fb_angmom(&(hier.hier[hier.hi[1]]), hier.nstar, Li);
	fb_angmomint(&(hier.hier[hier.hi[1]]), hier.nstar, Lint);
	for (j=0; j<3; j++) {
		Li[j] += Lint[j];
	}

	/* integrate along */
	fb_dprintf("calling fewbody()...\n");
	
	/* call fewbody! */
	retval = fewbody(input, units, &hier, &t, rng);

	/* print information to screen */
	fprintf(stderr, "OUTCOME:\n");
	if (retval.retval == 1) {
		fprintf(stderr, "  encounter complete:  t=%.6g (%.6g yr)  %s  (%s)\n\n",
			t, t * units.t/FB_CONST_YR,
			fb_sprint_hier(hier, string1),
			fb_sprint_hier_hr(hier, string2));
	} else {
		fprintf(stderr, "  encounter NOT complete:  t=%.6g (%.6g yr)  %s  (%s)\n\n",
			t, t * units.t/FB_CONST_YR,
			fb_sprint_hier(hier, string1),
			fb_sprint_hier_hr(hier, string2));
	}

	fb_dprintf("there were %ld integration steps\n", retval.count);
	fb_dprintf("fb_classify() was called %ld times\n", retval.iclassify);
	
	fprintf(stderr, "FINAL:\n");
	fprintf(stderr, "  t_final=%.6g (%.6g yr)  t_cpu=%.6g s\n", \
		t, t*units.t/FB_CONST_YR, retval.tcpu);

	fprintf(stderr, "  L0=%.6g  DeltaL/L0=%.6g  DeltaL=%.6g\n", fb_mod(Li), retval.DeltaLfrac, retval.DeltaL);
	fprintf(stderr, "  E0=%.6g  DeltaE/E0=%.6g  DeltaE=%.6g\n", Ei, retval.DeltaEfrac, retval.DeltaE);
	fprintf(stderr, "  Rmin=%.6g (%.6g RSUN)  Rmin_i=%d  Rmin_j=%d\n", \
		retval.Rmin, retval.Rmin*units.l/FB_CONST_RSUN, retval.Rmin_i, retval.Rmin_j);
	fprintf(stderr, "  Nosc=%d (%s)\n", retval.Nosc, (retval.Nosc>=1?"resonance":"non-resonance"));

	
	/* free GSL stuff */
	gsl_rng_free(rng);

	/* free our own stuff */
	fb_free_hier(hier);

	/* done! */
	return(0);
}
