/* -*- linux-c -*- */
/* binsingle.c

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
#include "binsingle.h"

/* print the usage */
void print_usage(FILE *stream)
{
	fprintf(stream, "USAGE:\n");
	fprintf(stream, "  binsingle [options...]\n");
	fprintf(stream, "\n");
	fprintf(stream, "OPTIONS:\n");
	fprintf(stream, "  -m --m0 <m0/MSUN>            : set mass of single star [%.6g]\n", FB_M0/FB_CONST_MSUN);
	fprintf(stream, "  -n --m10 <m10/MSUN>          : set mass of star 0 of binary [%.6g]\n", FB_M10/FB_CONST_MSUN);
	fprintf(stream, "  -o --m11 <m11/MSUN>          : set mass of star 1 of binary [%.6g]\n", FB_M11/FB_CONST_MSUN);
	fprintf(stream, "  -r --r0 <r0/RSUN>            : set radius of single star [%.6g]\n", FB_R0/FB_CONST_RSUN);
	fprintf(stream, "  -g --r10 <r10/RSUN>          : set radius of star 0 of binary [%.6g]\n", FB_R10/FB_CONST_RSUN);
	fprintf(stream, "  -i --r11 <r11/RSUN>          : set radius of star 1 of binary [%.6g]\n", FB_R11/FB_CONST_RSUN);
	fprintf(stream, "  -a --a1 <a1/AU>              : set semimajor axis of binary [%.6g]\n", FB_A1/FB_CONST_AU);
	fprintf(stream, "  -e --e1 <e1>                 : set eccentricity of binary 0 [%.6g]\n", FB_E1);
	fprintf(stream, "  -v --vinf <vinf/v_crit>      : set velocity at infinity [%.6g]\n", FB_VINF);
	fprintf(stream, "  -b --b <b/a1>                : set impact parameter [%.6g]\n", FB_B);
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
	fprintf(stream, "  -W --waves                   : prints the second-time dervative of the mass quadrupole moment [%d]\n",FB_QUAD);
	fprintf(stream, "  -V --version                 : print version info\n");
	fprintf(stream, "  -h --help                    : display this help text\n");
}

/* calculate the units used */
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

/* HVSS: calculates speed at infinity for top-level objects after encounter */
int calc_vfin(fb_hier_t *hier)
{
    int i;
    double pe, ke, ek_frac;

    /* calculates outer energies (outer = among the top-level objects) */
    pe = fb_outerpetot(hier->obj, hier->nobj);
    ke = fb_outerketot(hier->obj, hier->nobj);

    /* scaling factor for speeds at infinity from present state */
    ek_frac = sqrt((pe+ke)/ke);

    /* runs through top-level objects, updating v_post */
    for (i=0; i<hier->nobj; i++) {
        hier->obj[i]->vfin = ek_frac*fb_mod(hier->obj[i]->v);
    }

    return(0);
}

/* HVSS: print the product classification (bhbh+s, bhs+s, etc.) */
int calc_type_f(fb_hier_t hier, fb_ret_t *retval)
{
    int i, type_f, single_star, binary, bin_i;

    /* check if any single stars or binaries exist */
    single_star=0;
    binary = 0;
    for (i=0; i<hier.nobj; i++) {
        if (hier.obj[i]->n==1) {
		    if (hier.obj[i]->k_type!=14) {
                single_star = 1;
            }
        } else if (hier.obj[i]->n==2) {
            binary = 1;
            bin_i = i;
        }
    }

    /* filter using above values */
    if (!single_star) {
        type_f = 0;
    } else if (!binary) {
        if (hier.nstar==2) {
            type_f = 5;
        } else if (hier.nstar==3) {
            type_f = 6;
        }
    } else {
        type_f = 1;
        for (i=0; i<hier.obj[bin_i]->n; i++) {
            if (hier.obj[bin_i]->obj[i]->k_type!=14) {
                type_f = 3;
            }
        }
    }

    retval->type_f = type_f;

    return(0);
}

/* the main attraction */
int main(int argc, char *argv[])
{
	int i, j, k;
	unsigned long int seed;
	double m0, m10, m11, r0, r10, r11, a1, e1;
	double rtid, vinf, b, m1, M, mu, Ei, E, Lint[3], Li[3], l0[3], l1[3], L[3], r[3], t;
	double chi;
	int bid, sid;
	fb_hier_t hier;
	fb_input_t input;
	fb_ret_t retval;
	fb_units_t units;
	char string1[FB_MAX_STRING_LENGTH], string2[FB_MAX_STRING_LENGTH];
	gsl_rng *rng;
	const gsl_rng_type *rng_type=gsl_rng_mt19937;
	const char *short_opts = "m:n:o:r:g:i:K:L:M:a:e:C:v:b:t:D:c:A:R:N:z:y:x:P:Q:S:T:U:k:s:dVh";
	const struct option long_opts[] = {
		{"m0", required_argument, NULL, 'm'},
		{"m10", required_argument, NULL, 'n'},
		{"m11", required_argument, NULL, 'o'},
		{"r0", required_argument, NULL, 'r'},
		{"r10", required_argument, NULL, 'g'},
		{"r11", required_argument, NULL, 'i'},
		{"k0", required_argument, NULL, 'K'},
		{"k10", required_argument, NULL, 'L'},
		{"k11", required_argument, NULL, 'M'},
		{"a1", required_argument, NULL, 'a'},
		{"e1", required_argument, NULL, 'e'},
		{"chi", required_argument, NULL, 'C'},
		{"vinf", required_argument, NULL, 'v'},
		{"b", required_argument, NULL, 'b'},
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
		{"waves", no_argument, NULL, 'W'},
		{"version", no_argument, NULL, 'V'},
		{"help", no_argument, NULL, 'h'},
		{NULL, 0, NULL, 0}

	};

    /* HVSS; note that k0, k10, and k11 flags above are all HVSS additions */
    int k0, k10, k11;
    double a_fin, e_fin, Peo, Keo, Lbin[3], thetai;

	/* set parameters to default values */
	m0 = FB_M0;
	m10 = FB_M10;
	m11 = FB_M11;
	r0 = FB_R0;
	r10 = FB_R10;
	r11 = FB_R11;
	a1 = FB_A1;
	e1 = FB_E1;
	vinf = FB_VINF;
	b = FB_B;
	chi = -1.;
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
	input.fb_quad = FB_QUAD;
	seed = FB_SEED;
	fb_debug = FB_DEBUG;
    /* HVSS */
    k0 = 0;
    k10 = 0;
    k11 = 0;

	suppression=0.;
	J_pn=0;
	E_pn=0;
	e_pn=0;
	v_kick=0;
 	e_newt=0;

    /* HVSS: enables TDE criterion */
    input.BHNS_TDE_FLAG = 1;
	
	while ((i = getopt_long(argc, argv, short_opts, long_opts, NULL)) != -1) {
		switch (i) {
		case 'm':
			m0 = atof(optarg) * FB_CONST_MSUN;
			break;
		case 'n':
			m10 = atof(optarg) * FB_CONST_MSUN;
			break;
		case 'o':
			m11 = atof(optarg) * FB_CONST_MSUN;
			break;
		case 'r':
			r0 = atof(optarg) * FB_CONST_RSUN;
			break;
		case 'g':
			r10 = atof(optarg) * FB_CONST_RSUN;
			break;
		case 'i':
			r11 = atof(optarg) * FB_CONST_RSUN;
			break;
		case 'K':
			k0 = atoi(optarg);
			break;
		case 'L':
			k10 = atoi(optarg);
			break;
		case 'M':
			k11 = atoi(optarg);
			break;
		case 'a':
			a1 = atof(optarg) * FB_CONST_AU;
			break;
		case 'e':
			e1 = atof(optarg);
			if (e1 >= 1.0) {
				fprintf(stderr, "e0 must be less than 1\n");
				return(1);
			}
			break;
		case 'v':
			vinf = atof(optarg);
			if (vinf < 0.0) {
				fprintf(stderr, "vinf must be non-negative\n");
				return(1);
			}
			break;
		case 'b':
			b = atof(optarg);
			if (b < 0.0) {
				fprintf(stderr, "b must be non-negative\n");
				return(1);
			}
			break;
		case 't':
			input.tstop = atof(optarg);
			break;
		case 'D':
			input.Dflag = 1;
			input.dt = atof(optarg);
			if (input.dt < 0) input.Dflag = 2;
			break;
		case 'c':
			input.tcpustop = atof(optarg);
			break;
		case 'C':
			chi = atof(optarg);
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
		case 'W':
			input.fb_quad = 1;
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
	hier.nstarinit = 3;
	hier.nstar = 3;
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
	fprintf(stderr, "  m0=%.6g MSUN  r0=%.6g RSUN\n", m0/FB_CONST_MSUN, r0/FB_CONST_RSUN);
	fprintf(stderr, "  a1=%.6g AU  e1=%.6g  m10=%.6g MSUN  m11=%.6g MSUN  r10=%.6g RSUN  r11=%.6g RSUN\n", \
		a1/FB_CONST_AU, e1, m10/FB_CONST_MSUN, m11/FB_CONST_MSUN, r10/FB_CONST_RSUN, r11/FB_CONST_RSUN);
	fprintf(stderr, "  vinf=%.6g  b=%.6g  tstop=%.6g  tcpustop=%.6g\n", \
		vinf, b, input.tstop, input.tcpustop);
	fprintf(stderr, "  tidaltol=%.6g  speedtol=%.6g  abs_acc=%.6g  rel_acc=%.6g  ncount=%d  fexp=%.6g\n", \
		input.tidaltol, input.speedtol, input.absacc, input.relacc, input.ncount, input.fexp);
	fprintf(stderr, "  PN1=%d  PN2=%d  PN25=%d  PN3=%d  PN35=%d\n\n", \
		input.PN1, input.PN2, input.PN25, input.PN3, input.PN35);

	/* initialize GSL rng */
	gsl_rng_env_setup();
	rng = gsl_rng_alloc(rng_type);
	gsl_rng_set(rng, seed);

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

	hier.hier[hier.hi[1]+0].R = r0;
	hier.hier[hier.hi[1]+1].R = r10;
	hier.hier[hier.hi[1]+2].R = r11;

	hier.hier[hier.hi[1]+0].m = m0;
	hier.hier[hier.hi[1]+1].m = m10;
	hier.hier[hier.hi[1]+2].m = m11;

	hier.hier[hier.hi[1]+0].chi = chi;
	hier.hier[hier.hi[1]+1].chi = chi;
	hier.hier[hier.hi[1]+2].chi = chi;

	hier.hier[hier.hi[2]+0].m = m10 + m11;

	hier.hier[hier.hi[2]+0].a = a1;
	hier.hier[hier.hi[2]+0].e = e1;

    /* HVSS: labels objects with ktypes, default 0; label binary with -1 */
	hier.hier[hier.hi[1]+0].k_type = k0;
	hier.hier[hier.hi[1]+1].k_type = k10;
	hier.hier[hier.hi[1]+2].k_type = k11;    

	/* HVSS: updates BH radii to be 5 * Schwartzchild radius */
	for (i=0; i<hier.nstar; i++) {
	    if (hier.hier[hier.hi[1]+i].k_type == 14) {
		hier.hier[hier.hi[1]+i].R = 2 * FB_REFF_BH * FB_CONST_G * hier.hier[hier.hi[1]+i].m  / (FB_CONST_C*FB_CONST_C);
	    }	    
	}
	
	hier.obj[0] = &(hier.hier[hier.hi[1]+0]);
	hier.obj[1] = &(hier.hier[hier.hi[2]+0]);
	hier.obj[2] = NULL;

	/* get the units and normalize */
	calc_units(hier.obj, &units);
	fb_normalize(&hier, units);

	fprintf(stderr, "UNITS:\n");
	fprintf(stderr, "  v=v_crit=%.6g km/s  l=%.6g AU  t=t_dyn=%.6g yr\n", \
		units.v/1.0e5, units.l/FB_CONST_AU, units.t/FB_CONST_YR);
	fprintf(stderr, "  M=%.6g M_sun  E=%.6g erg\n\n", units.m/FB_CONST_MSUN, units.E);

	/* move hierarchies analytically in from infinity along hyperbolic orbit */
	fb_dprintf("moving hierarchies analytically in from infinity...\n");

	m0 = hier.obj[0]->m;
	m1 = hier.obj[1]->m;
	M = m0 + m1;
	mu = m0 * m1 / M;

	Ei = 0.5 * mu * fb_sqr(vinf);

	a1 = hier.obj[1]->a;
	e1 = hier.obj[1]->e;
	m10 = hier.obj[1]->obj[0]->m;
	m11 = hier.obj[1]->obj[1]->m;

	rtid = pow(2.0*(m0+m1)/(m1*input.tidaltol), 1.0/3.0) * a1 * (1.0+e1);

	fb_init_scattering(hier.obj, vinf, b, rtid);
	
	/* and check to see that we conserved energy and angular momentum */
	fb_cross(hier.obj[0]->x, hier.obj[0]->v, l0);
	fb_cross(hier.obj[1]->x, hier.obj[1]->v, l1);
	
	for (i=0; i<3; i++) {
		L[i] = (m0 * l0[i] + m1 * l1[i]);
		r[i] = hier.obj[1]->x[i] - hier.obj[0]->x[i];
	}

	E = - m0 * m1 / fb_mod(r) + 0.5 * (m0 * fb_dot(hier.obj[0]->v, hier.obj[0]->v) + \
					m1 * fb_dot(hier.obj[1]->v, hier.obj[1]->v));

	fb_dprintf("L0=%.6g DeltaL/L0=%.6g DeltaL=%.6g\n", mu*b*vinf, fb_mod(L)/(mu*b*vinf)-1.0, fb_mod(L)-mu*b*vinf);
	fb_dprintf("E0=%.6g DeltaE/E0=%.6g DeltaE=%.6g\n\n", Ei, E/Ei-1.0, E-Ei);

	/* trickle down the binary properties, then back up */
	fb_dprintf("obj[%d]->a=%e\n", 1, hier.obj[1]->a);
	fb_dprintf("obj[%d]->e=%e\n", 1, hier.obj[1]->e);
	fb_dprintf("obj[%d]->m=%e\n", 1, hier.obj[1]->m);
	fb_randorient(&(hier.hier[hier.hi[2]+0]), rng);
	fb_downsync(&(hier.hier[hier.hi[2]+0]), t);
	fb_upsync(&(hier.hier[hier.hi[2]+0]), t);
	fb_dprintf("obj[%d]->a=%e\n", 1, hier.obj[1]->a);
	fb_dprintf("obj[%d]->e=%e\n", 1, hier.obj[1]->e);
	fb_dprintf("obj[%d]->m=%e\n", 1, hier.obj[1]->m);
	
	/* store the initial energy and angular momentum*/
	Ei = fb_petot(&(hier.hier[hier.hi[1]]), hier.nstar) + fb_ketot(&(hier.hier[hier.hi[1]]), hier.nstar) +
		fb_einttot(&(hier.hier[hier.hi[1]]), hier.nstar);
	fb_angmom(&(hier.hier[hier.hi[1]]), hier.nstar, Li);
	fb_angmomint(&(hier.hier[hier.hi[1]]), hier.nstar, Lint);
	for (j=0; j<3; j++) {
		Li[j] += Lint[j];
	}

	/* Ltest: use internal utility to save angular momentum */
	fb_angmom_bin(&(hier.hier[hier.hi[2]+0]), hier.hier[hier.hi[2]+0].n, Lbin);

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
	fprintf(stderr, "  E0=%.6g  DeltaE/E0=%.6g  DeltaE=%.6g  DeltaE_GW=%.6g\n", Ei, retval.DeltaEfrac, retval.DeltaE, retval.DeltaE_GW/Ei);
	fprintf(stderr, "  Rmin=%.6g (%.6g RSUN)  Rmin_i=%d  Rmin_j=%d\n", \
		retval.Rmin, retval.Rmin*units.l/FB_CONST_RSUN, retval.Rmin_i, retval.Rmin_j);
	fprintf(stderr, "  Nosc=%d (%s)\n", retval.Nosc, (retval.Nosc>=1?"resonance":"non-resonance"));

    /* HVSS: find final binary semi-major axis, 0 if no binary exists */
    a_fin = 0;
    e_fin = 0;
    for (i=0; i<hier.nobj; i++) {
        if (hier.obj[i]->n==2) {
            a_fin = hier.obj[i]->a;
	    e_fin = hier.obj[i]->e;
        }
    }

    /* HVSS: calls calc_vfin function, calculating speeds at infinity for all top-level objects */
    calc_vfin(&hier);

    /* HVSS: calculates classification index for outcome */
    calc_type_f(hier, &retval);

    /* print the things for python */
    fprintf(stdout, " %d", retval.type_f);
    fprintf(stdout, " %g", units.v/1.0e5); // /1.0e5 converts from cm/s to km/s
    fprintf(stdout, " %g %g", a_fin, e_fin);
    fprintf(stdout, " %g %g %g", Li[0], Li[1], Li[2]);
    fprintf(stdout, " %g %g %g", Lbin[0], Lbin[1], Lbin[2]);
    if (hier.nobj == 3) {
        fprintf(stdout, " %g %d %g %d", hier.obj[0]->vfin, hier.obj[0]->k_type, hier.obj[0]->Rmin, hier.obj[0]->Rmin_j);
        fprintf(stdout, " %g %d %g %d", hier.obj[1]->vfin, hier.obj[1]->k_type, hier.obj[1]->Rmin, hier.obj[1]->Rmin_j);
        fprintf(stdout, " %g %d %g %d", hier.obj[2]->vfin, hier.obj[2]->k_type, hier.obj[2]->Rmin, hier.obj[2]->Rmin_j);
    } else if (hier.nobj == 2) {
        fprintf(stdout, " %g %d %g %d", hier.obj[0]->vfin, hier.obj[0]->k_type, hier.obj[0]->Rmin, hier.obj[0]->Rmin_j);
        fprintf(stdout, " %g %d %g %d", hier.obj[1]->vfin, hier.obj[1]->k_type, hier.obj[1]->Rmin, hier.obj[1]->Rmin_j);
        fprintf(stdout, " %g %d %g %d", 0., -1, 0., 0);
    } else if (hier.nobj == 1) {
        fprintf(stdout, " %g %d %g %d", hier.obj[0]->vfin, hier.obj[0]->k_type, hier.obj[0]->Rmin, hier.obj[0]->Rmin_j);
        fprintf(stdout, " %g %d %g %d", 0., -1, 0., 0);
        fprintf(stdout, " %g %d %g %d", 0., -1, 0., 0);
    }
    fprintf(stdout, "\n");

	/* free GSL stuff */
	gsl_rng_free(rng);

	/* free our own stuff */
	fb_free_hier(hier);

	/* done! */
	return(0);
}

/*-----------------------------------------------------------------------------------------
-----------------------------------Graveyard-----------------------------------------------
-----------------------------------------------------------------------------------------*/

///* print the velocities of each star after the encounter */
//int fprint_v(fb_hier_t hier)
//{
//
//    int i, j, k;
//
//    fprintf(stderr, "Starting iteration...\n");
//
//    for (i=0; i<hier.nstar; i++) {
//        fprintf(stdout, "vs%d: %10.6g %10.6g %10.6g\n", i, hier.hier[hier.hi[1]+i].v[0], hier.hier[hier.hi[1]+i].v[1], hier.hier[hier.hi[1]+i].v[2]);
//    }
//
//    for (i=0; i<hier.nobj; i++) {
//        if (hier.obj[i]->n == 2) {
//            fprintf(stdout, "vb%d: %10.6g %10.6g %10.6g \n", i, hier.obj[i]->v[0], hier.obj[i]->v[1], hier.obj[i]->v[2]);
//        } else if (hier.obj[i]->n == 3) {
//            fprintf(stdout, "vt%d: %10.6g %10.6g %10.6g \n", i, hier.obj[i]->v[0], hier.obj[i]->v[1], hier.obj[i]->v[2]);
//        }
//    }
//
//    /* code here flattens hierarchy and prints velocities, for checking
//    fprintf(stderr, "Flattening hierarchy...\n");
//    fb_init_hier(&hier);
//
//    for (i=0; i<hier.nobj; i++) {
//        fprintf(stdout, "Velocity for Object %d: %.6g %.6g %.6g\n", i, hier.obj[i]->v[0], hier.obj[i]->v[1], hier.obj[i]->v[2]);
//    } */
//
//    return(0);
//}
//
///* print the v_infs of each star after the encounter */
//int fprint_v_inf(fb_hier_t hier, double pe, double ke)
//{
//
//    int i;
//    double ek_frac;
//
//    ek_frac = sqrt((pe+ke)/ke);
//
//    fprintf(stderr, "Starting iteration...\n");
//
//    for (i=0; i<hier.nobj; i++) {
//        if (hier.obj[i]->n == 1) {
//            fprintf(stdout, "vs%d: %10.6g %10.6g %10.6g \n", i, ek_frac*hier.obj[i]->v[0], ek_frac*hier.obj[i]->v[1], ek_frac*hier.obj[i]->v[2]);
//        } else if (hier.obj[i]->n == 2) {
//            fprintf(stdout, "vb%d: %10.6g %10.6g %10.6g \n", i, ek_frac*hier.obj[i]->v[0], ek_frac*hier.obj[i]->v[1], ek_frac*hier.obj[i]->v[2]);
//        } else if (hier.obj[i]->n == 3) {
//            fprintf(stdout, "vt%d: %10.6g %10.6g %10.6g \n", i, ek_frac*hier.obj[i]->v[0], ek_frac*hier.obj[i]->v[1], ek_frac*hier.obj[i]->v[2]);
//        }
//    }
//
//    return(0);
//}

///* function to print k_types of objects, with recusion for multiples */
//char *fb_obj_to_k(fb_obj_t *obj) {
//
//    char string[FB_MAX_STRING_LENGTH];
//
//    fprintf(stdout, "k_type_obj: %d\n", obj->k_type);
//
//    if (obj->n == 1) {
//        return(obj->k_type);
//	    //snprintf(string, FB_MAX_STRING_LENGTH-strlen(string), " %s", obj->k_type);
//	} else {
//        snprintf(string, FB_MAX_STRING_LENGTH-strlen(string), " [%s %s]", fb_obj_to_k(obj->obj[0]), fb_obj_to_k(obj->obj[1]));
//        return(string);
//        
//    }
//}
//
///* print the hierarchy information, but with k_types*/
//char *fb_sprint_hier_k(fb_hier_t hier)
//{
//	int i;
//    char string[FB_MAX_STRING_LENGTH];
//
//	for (i=0; i<hier.nobj; i++) {
//        fprintf(stdout, "k_type: %d\n", hier.obj[i]->k_type);
//        //snprintf(string, FB_MAX_STRING_LENGTH-strlen(string), " %s", fb_obj_to_k(hier.obj[i]));
//        strcat(string, fb_obj_to_k(hier.obj[i]));
//    }
//	
//	return(string);
//}
///* print the v_inf max of the encounter, with the type of the respective object */
//int fprint_v_inf_max(fb_hier_t hier, double pe, double ke)
//{
//
//    int i, k_max;
//    double ek_frac, v_inf, v_inf_max;
//
//    /* set initial values */
//    v_inf_max = 0.;
//    k_max = 1;
//
//    ek_frac = sqrt((pe+ke)/ke);
//
//    fprintf(stderr, "Starting iteration...\n");
//
//    for (i=0; i<hier.nstar; i++) {
//        v_inf = ek_frac*fb_mod(hier.hier[hier.hi[1]+i].v);
//
//        if (v_inf > v_inf_max) {
//            v_inf_max = v_inf;
//            k_max = hier.hier[hier.hi[1]+i].k_type;
//        }
//    }
//            
//    fprintf(stdout, "v_max %10.6g k_max %d\n", v_inf_max, k_max);
//
//    return(0);
//}
