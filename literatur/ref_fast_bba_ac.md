# REFERANS: Fast beam-based alignment using ac excitations (PRAB 23, 012802)

> Kaynak PDF'ten pdftotext ile çıkarıldı (özgünlük karşılaştırması için referans).

---

                      PHYSICAL REVIEW ACCELERATORS AND BEAMS 23, 012802 (2020)



                                   Fast beam-based alignment using ac excitations
                                        Zeus Martí ,* Gabriele Benedetti, and Ubaldo Iriso
                             ALBA Synchrotron, C. de la Llum 2-26, 08290 Cerdanyola del Vallès, Spain

                                                            Andrea Franchi
                                         ESRF, CS 40220, 38043 Grenoble CEDEX 9, France

                                           (Received 1 October 2019; published 22 January 2020)

                   Standard quadrupole beam-based alignment (BBA) techniques rely on orbit data and on the sequential
                variation of quadrupole and orbit corrector magnets (OCM). This results in time-consuming measurements
                of the order of several hours in most circular accelerators. Fast (10 kHz) beam position monitors (BPM) and
                OCMs with ac power supplies are routinely used in modern synchrotron light sources to drive fast orbit
                feedback systems. In this paper we show how they can be employed also to dramatically reduce the time for
                any quadrupole BBA to several minutes only, ensuring the same level of accuracy and precision. Moreover,
                conversely to the standard BBA, the new procedure accounts automatically for any level of betatron
                coupling, BPM roll and OCM tilt. In the case of the ALBA 3rd generation light source, the time for a
                complete measurement dropped from 5 hours to 10 minutes, a reduction by a factor 30. As further
                extension of this novel approach, an even faster skew quadrupole BBA was demonstrated in ALBA for the
                first time, taking advantage of the additional ac modulation of the skew quadrupole field. Results from this
                fully ac measurement are compared with those obtained via a traditional dc scan of the skew quadrupole.

                DOI: 10.1103/PhysRevAccelBeams.23.012802


                       I. INTRODUCTION                                  does not provide directly the BPM-to-quad offsets, though
                                                                        they can be interpolated from the measured ones. This
   Aligning the beam centroid to the center of the magnetic
                                                                        method gives a direct indication of how well centered is the
elements is an efficient way to minimize unwanted multipole
                                                                        beam, but not how much the individual quadrupoles or the
feed-down effects in particle accelerators. A beam crossing
                                                                        BPMs have to be realigned. To achieve that the model has
magnets off axis experiences orbit distortion and alteration
                                                                        to be used again.
of the dispersion function (in quadrupoles via dipole feed-
                                                                           The BPM-to-quad technique does not rely on the
down), as well as linear optics errors and coupling (in
                                                                        accelerator optics model but needs to scan the strength
sextupoles via normal and skew quadrupole feed-down).
                                                                        of one or several OCMs at every quadrupole change. This is
The beam-based alignment (BBA) facilitates the work of
                                                                        usually much more time consuming and it may give
corrector magnets to minimize these errors and reduces the
                                                                        inaccurate results if the beam orbit has a pronounced angle
deviation between the real machine and the accelerator
                                                                        at the quadrupole. The aim of the method is to align the
model. Two different quadrupole BBA approaches are
                                                                        BPMs, which are prone to electronic induced offsets, to the
described in the literature. In the first, the observable is
                                                                        magnets layout, which is assumed to be more stable over
the offset between beam trajectory and quadrupole axis
                                                                        time. Besides, the BPMs are usually attached to the vacuum
[1–4]. In the second, the observable is the beam position at
                                                                        chamber and need to be realigned after interventions.
the BPM when it passes through the axis of the closest
                                                                        Magnets are misaligned too, but this lays outside the scope
quadrupole [5–8]. The first technique is known as beam-to-
                                                                        of this method. In synchrotron light sources the quadrupole
quad while the second is referred to as BPM-to-quad.
                                                                        scan is typically performed step by step in a dc way and
   The beam-to-quad technique does not require OCM
                                                                        slow acquisition (10 Hz) orbit position data are analyzed,
scans but is model dependent: Tunes and optical functions
                                                                        whereas the BBA of quadrupoles in the interaction regions
are needed to infer the beam-to-quad offset. Moreover, it
                                                                        of colliders is usually carried out via a continuous quadru-
                                                                        pole gradient modulation and the harmonic analysis of turn-
  *
      zeus@cells.es                                                     by-turn BPM data. In the literature the latter approach is
                                                                        referred to as k-modulation [5] and requires quadrupole
Published by the American Physical Society under the terms of           power supplies with ac capabilities, presently not available
the Creative Commons Attribution 4.0 International license.
Further distribution of this work must maintain attribution to          in ALBA. Therefore, standard BPM-to-quad BBA used to
the author(s) and the published article’s title, journal citation,      be carried out in its storage ring via slow acquisition orbit
and DOI.                                                                position data and dc changes of both quadrupoles and

2469-9888=20=23(1)=012802(23)                                   012802-1               Published by the American Physical Society
MARTÍ, BENEDETTI, IRISO, and FRANCHI                                    PHYS. REV. ACCEL. BEAMS 23, 012802 (2020)

OCMs [9]. A complete measurement over 120 BPMs in                 quadrupole magnets under various assumptions and con-
both planes takes usually 5 hours. An overview of different       ditions. In Sec. II A we consider exciting horizontal and
BBA techniques and implementations can be found in [10].          vertical OCMs separately while in Sec. II B we consider a
   In this paper we present a fast BBA (FBBA) BPM-to-quad         simultaneous orbit modulation in the two planes, though
method based on the parallel ac modulation of several             still ignoring coupling effects. A complete treatment
OCMs at different frequencies and on the harmonic                 including betatron coupling, BPM roll, OCM tilt and
analysis of orbit data sampled at high frequency (10 kHz).        dual-plane orbit modulation is presented in Sec. II C.
A complete FBBA requires 10 minutes only, which repre-            While the first two approaches are valid for normal
sents a reduction by a factor 30 compared to the standard         quadrupoles only, the last is more general and is applicable
5 hours. It is worthwhile stressing the fact that the FBBA is a   to both normal and skew quadrupoles. A further extension
different scheme compared to the k-modulation, where the          to make use of an ac modulation of the quadrupole itself
ac modulation is performed on the quadrupoles, not on the         (either normal or skew) on top of the ac modulation of
steerers, and the harmonic analysis is performed on the beam      the OCMs is also presented. If implemented, this fully
trajectory (turn-by-turn data), not on the orbit.                 ac variant would further reduce the FBBA time by an
   A fully ac FBBA with harmonic modulation of the                additional factor two.
magnet to be centered was proved for the first time at
ALBA by centering the beam into skew quadrupoles,
                                                                                 A. Single-plane ac excitation
which are trim coils mounted on the main sextupoles
and feature ac capabilities. Besides the fact of being an           In the linear regime and in the presence of a time-varying
even faster operation, this analysis aligns automatically also    OCM setting, vj ðtÞ, the vertical closed-orbit readings yl ðtÞ
the sextupole magnets onto which the skew quadrupole              and yk ðtÞ at two BPMs l and k read
coils are installed, with benefits to the machine optics.
   A fast acquisition archiver (FA) providing synchronous                       yl ðtÞ − y0l ¼ Ryy
                                                                                                lj ðvj ðtÞ − v0 Þ;
BPM data at 10 kHz rate originally developed at Diamond
[11] was installed in ALBA [12] and other synchrotron light                     yk ðtÞ − y0k ¼ Ryy
                                                                                                kj ðvj ðtÞ − v0 Þ;            ð1Þ
sources since 2011. OCMs can be then excited in parallel in
both planes and at different frequencies. The motion induced      where t represents discrete, equally spaced time values
by any steerer in each plane can be isolated and analyzed
                                                                  (sampled at a 10 kHz rate in the case of ALBA), y0 is the
separately from the harmonic analysis of FA data during the
                                                                  BPM position reading when the OCM setting is v0 . As any
OCM modulation by looking at the spectral peaks generated
                                                                  other BPM-to-quad BBA implementation, this method
at those (known) frequencies. Even though FA data contain
                                                                  does not relies on the accelerator model knowledge so
position data synchronized among all 120 BPMs, this is not
                                                                  the effective orbit response matrix (ORM) Ryylj is unknown.
the case for the ac OCMs. Their currents are either not
archived (as in the case of ALBA) or not suitable for             In this paper we will assume that the BPM l is next to the
conversion into strengths, also their calibration factors are     varied quadrupole as sketched in Fig. 1. The OCM j is
frequency-dependent and not known a priori. This requires         chosen to produce a large variation of the orbit at BPM l.
some preliminary mathematical gymnastics of the original          Hence, although we do not assume any specific value for
BBA formulas to remove any dependence of the observables          Ryy
                                                                    lj , it should not have a small value at BPM l.
on the OCM strength. The formalism developed to this end
accounts also for the presence of an arbitrarily large betatron
coupling, OCM tilt, and BPM roll.
   The paper is structured as follows: the FBBA is presented
in Sec. II. In Sec. III the experimental set-up is described.
The error analysis based on the alignment tolerances and
the expected accuracy are evaluated in Sec. IV. In Sec. V
experimental results of the new methodology applied to
ALBA are presented and compared with the standard BBA.
Some conclusions are drawn in Sec. VI, whereas the
mathematical details of the proposed approach are presented
in Appendices A and B.

      II. QUADRUPOLE FAST BEAM-BASED
                ALIGNMENT                                         FIG. 1. Sketch of the beam closed orbit when passing through
                                                                  the center of the quadrupole (next to the BPM l) to be aligned.
  In this section we present different approaches to              In the example drawn vj ðtÞ ¼ v0 generates the closed orbit that
evaluate the BPM-to-quad offsets with respect to                  makes the beam pass through the quadrupole center.


                                                            012802-2
FAST BEAM-BASED ALIGNMENT …                                                 PHYS. REV. ACCEL. BEAMS 23, 012802 (2020)

  While a synchronous temporal sampling of the BPM
data is ensured by the FA, in the case of ALBA, the OCMs
data vj ðtÞ are not available, hence not observable.
Nevertheless, the two equations in the above system can
be combined to remove any dependence on the OCM
excitation, namely

                                 Ryy
              yl1 ðtÞ − y0l1 ¼          ðyk1 ðtÞ − y0k1 Þ:
                                  lj1
                                                              ð2Þ
                                 Ryy
                                  kj1

The above equation describes the linear correlation
between BPM data during a beam steering. The label 1
refers to the first quadrupole setting. Any change in the
quadrupole current ΔI modifies both the response matrix
elements, i.e., the slope in Eq. (2), and the orbit distortion if
the beam is not centered to the quadrupole under study.
These effects can be written as

                                 Ryy
              yl2 ðtÞ − y0l2 ¼          ðyk2 ðtÞ − y0k2 Þ;
                                  lj2
                                                              ð3Þ
                                 Ryy
                                  kj2                                FIG. 2. Example of measured only vertical plane FBBA. The
                                                                     BPM offsets leading to the on-axis crossing of the beam through
where the label 2 refers to the modified quadrupole strength         the quadrupole are inferred from the intercept of the two lines:
(or modified quadrupole current, I 0 þ ΔI). The new con-             before (blue) and after (red) the quadrupole change ΔI ¼ 2.5 A.
stant values can be written as y0l2 ¼ y0l1 þ Al and                  The OCM was excited at 6 Hz. The data shown here corresponds
y0k2 ¼ y0k1 þ Ak , where Al and Ak account for the orbit             to the case l ¼ 69.
distorted by the off-axis passage through the quadrupole.
Equation (3) can be then rewritten as                                the information of one BPM (BPM k) apart from the BPM
                                                                     closest to the quadrupole (BPM l), the estimated offset is
                         Ryy
          yl2 ðtÞ − yl1 ¼ yy ðyk2 ðtÞ − y0k1 Þ þ Alkj ;
                     0    lj2
                                                              ð4Þ    98  16 μm. The error bar is computed as the distance
                         Rkj2                                        along the axis of the BPM l from the intersection point
                                                                     when a separation between the two fit lines equals the BPM
where Alkj ¼ Al − Ryy            yy
                           lj2 =Rkj2 Ak represents an offset with    noise. The latter is evaluated from the fluctuations in the
respect to the line of Eq. (2). The intersection between two         BPM readings with unperturbed beam.
(or more) lines obtained with two (or more) quadrupole                  Accuracy and precision can be improved by using data
settings would correspond to the beam crossing the quadru-           from several BPMs k. Ideally, the two lines of Fig. 2 shall
pole on axis, since the orbit distortion would not depend            have sufficiently different slopes so to increase both the
on its strength. According to Eqs. (2) and (4), the only             accuracy, when determining the intersection, and the
condition for two lines to have the same value is when both          precision, when computing the error bar. Similarly, differ-
hand sides are zero, i.e., when Alkj ¼ 0 ⇒ y0l2 ¼ y0l1 ;             ent pairs of BPMs provide different accuracy and precision,
y0k2 ¼ y0k1 , yl ðtÞ ¼ y0l1 and yk ðtÞ ¼ y0k1 . The physical mean-   because of their different sensitivity to determine the offset,
                                                                     that is proportional to the change of the slopes in Eqs. (2)
ing of y0l1 and y0k1 in Eq. (2) is then nothing else than the                             Ryy     Ryy
BPM readings for which the beam crosses the quadrupole               and (3), namely Ryylj2 − Ryylj1 .
                                                                                            kj2    kj1
on axis, i.e., when the beam is aligned. If one of the                  For each quadrupole, the results presented here have
monitors is close to the quadrupole under scrutiny, e.g., the        been obtained first by selecting the most suitable OCM and
BPM l, y0l denotes the BPM-to-quad offset. The offset can            then by averaging the results over different pairs of BPMs.
be obtained not knowing the exact value of the response              Couples with error bars larger than three times the average
matrix Ryy  lj , which depends on the OCM and BPM                    error bar (i.e., with weak correlation) are discarded. In the
calibration factors. This technique, as well as the original         example of Fig. 2, by using the data from all the BPMs a
BBA, is completely model independent.                                simple average leads to an offset of 84  10 μm. Instead, if
   A graphical illustration of the method is given in Fig. 2         the above mentioned outliers are discarded, the final offset
where the two lines measured prior and after a change                is 92  3 μm. Figure 3 shows the offset estimation of BPM
in the quadrupole current, corresponding to Eqs. (2)                 No. 69 given by each one of the other 119 BPMs. The red 1
and (3), are displayed. In this particular case, using only          dots represent the discarded BPM results and correspond

                                                               012802-3
MARTÍ, BENEDETTI, IRISO, and FRANCHI                                         PHYS. REV. ACCEL. BEAMS 23, 012802 (2020)

                                                                       up before applying the method described in the previous
                                                                       section. Equation (2) is generalized to

                                                                                                        Rxx
                                                                                      x̃l ðtÞ − x0l ¼         ðx̃k ðtÞ − x0k Þ;
                                                                                                         lj
                                                                                                                                   ð5Þ
                                                                                                        Rxx
                                                                                                         kj


                                                                                                        Ryy
                                                                                      ỹl ðtÞ − y0l ¼         ðỹk ðtÞ − y0k Þ;
                                                                                                         lj
                                                                                                                                   ð6Þ
                                                                                                        Ryy
                                                                                                         kj


                                                                       where

                                                                                          x̃l ðtÞ ¼ xl;dc þ xl;fh ðtÞ;             ð7Þ

                                                                                          ỹl ðtÞ ¼ yl;dc þ yl;fv ðtÞ:             ð8Þ

                                                                       xdc and ydc denote the dc originated components of the FA
                                                                       data, whereas xfh ðtÞ and yfv ðtÞ represent their harmonics
                                                                       oscillating at the horizontal and vertical excitation frequen-
                                                                       cies, f h and f v , respectively. Any BPM response due to
FIG. 3. The upper plot shows the BPM l estimated offsets               other external sources is hence removed. Under the
obtained from the lines intersection using data from all the other
                                                                       assumption of operating the machine at low betatron
119 BPMs. The cases where the error bar is larger than three
times the average error bar are highlighted in red and removed         coupling, its impact can be ignored and the analysis of
from the final average. The lower plot shows the inverse of the        Sec. II A repeated on the signals x̃ and ỹ.
difference between the slopes in Eqs. (2) and (3) estimated from          Before showing an example of the multifrequency
the model. Notice that the removed BPMs (red dots) in the upper        FBBA, it shall be noticed that betatron coupling and the
plot coincide with those for which the such sensitivity factor is      finite sampling frequency (10 kHz) introduce some cross-
close to zero (the inverse factor has a peak) in the lower plot. The   talk among the BPM harmonics. Even in machines operat-
data shown here corresponds to the case BPM l ¼ 69.                    ing at low coupling, this interference may appear at those
                                                                       BPMs with weak response to OCM excitation. This issue is
well with peaks of the inverse of the sensitivity factor               addressed in Appendix B, where a numerical solution is
estimated from the model.                                              presented to correct the cross-talk. The cross talk depends
   It is generally assumed that the quadrupoles are                    on how close are the selected ac frequencies of each plane
better aligned and keep their alignment over time better               but there are other considerations that influence the
than the BPMs. In most cases, after each measurement,                  frequency selection and are addressed in Sec. III. In the
an artificial BPM numerical offset y0l1 is added to ensure             case of ALBA the most suitable frequencies were found
that the closed-orbit correction centers the beam onto the             to be 7 Hz in the horizontal plane and 6 Hz in the
quadrupole axis.                                                       vertical plane.
                                                                          As for the single-plane, single-frequency measurement
                                                                       introduced in the previous section, the intersection of the
   B. Multi-frequency and dual-plane ac excitation                     lines of Eq. (5) before and after the quadrupole change
   The analysis presented in the previous section does not             provides the BPM-to-quad offsets, x0l and x0k . The same
depend on the frequency of the ac OCM excitation. In order             exercise applied to the vertical data via Eq. (6) will
to fully exploit the ac nature of this approach and to speed           determine the vertical BPM-to-quad offsets.
up the overall measurement, quadrupole offset in both                     In Fig. 4 results from a (parallel) dual-plane excitation
planes can be obtained in a single measurement by a                    are displayed along with those from the equivalent (sequen-
simultaneous excitation of horizontal and vertical OCMs at             tial) single-plane excitations. In the top plot the raw vertical
different frequencies. The FA data stream will contain the             data sampled at 10 kHz (corresponding to Eq. (2) are
beam response to both excitations, and so a Fourier analysis           reported. The blue and red lines refer to the single-plane
can be performed to decouple the two responses. In Eq. (2)             excitation. The green and black curves are from the dual-
the yl ðtÞ signal contains the orbit response both from the dc         plane excitation, whose beating is generated by betatron
OCM strength and the (not observable) ac OCM excitation                coupling mixing the two modulations. The ỹ data of Eq. (6)
vj ðtÞ. When the beam is excited by two OCMs (horizontal               obtained from the harmonic analysis of those curves are
and vertical) at different frequencies, both dc and ac                 reported in the bottom plot, showing how the two lines
originated components need to be isolated and summed                   obtained from the dual-frequency OCM excitation match

                                                                012802-4
FAST BEAM-BASED ALIGNMENT …                                               PHYS. REV. ACCEL. BEAMS 23, 012802 (2020)

                                                                    By doing so, the ratio between the above two equations
                                                                    evaluated at two BPMs does not yield any longer the
                                                                    observable linear relation of Eq. (2). A different (and novel)
                                                                    mathematical approach is then required.
                                                                       Equation (9) is general enough to be valid for any
                                                                    amount of OCM tilts. The inclusion of Rxy and Ryx can
                                                                    absorb coupling effects of any kind. A brief proof is
                                                                    presented in Appendix Sec. A 1 of Appendix A.
                                                                       The inclusion of betatron coupling allows a straightfor-
                                                                    ward extension of the FBBA to skew quadrupoles. This is of
                                                                    particular interest in synchrotron light sources since these
                                                                    magnets are usually trim coils mounted on the yokes of
                                                                    sextupoles: aligning the beam at a skew quadrupole would
                                                                    hence correspond to a sextupole BBA, which benefits the
                                                                    machine linear optics and betatron coupling. Moreover,
                                                                    since trim coils usually feature ac functionalities, the FBBA
                                                                    can be extended to a fully ac approach by varying their
                                                                    strengths sinusoidally instead of repeating the measurement
                                                                    at two different strengths, hence halving the measurement
                                                                    time. The frequency of the skew quadrupole excitation is
                                                                    chosen so to not interfere with the ones of the OCMs.
                                                                       In this sections only the final results are presented and
                                                                    discussed, while all proofs and details can be found in
                                                                    Appendix A. There, a general formula evaluating the BPM-
                                                                    to-quad offset l is derived:

                                                                    x0l ¼ ℜfhxl j0ig þ Sfhxl jf h igMh þ Sfhxl jf v igMv
                                                                    y0l ¼ ℜfhyl j0ig þ Sfhyl jf v igMv þ Sfhyl jf h igMh ;   ð10Þ
FIG. 4. Top plot: correlation between two BPMs vertical raw
data sampled at 10 kHz before (green dots) and after (black dots)   where the coefficients M are observable independent on
a quadrupole change during the simultaneous excitation of a         the type of quadrupole (be it normal or skew), though they
horizontal (at 7 Hz) and a vertical (at 6 Hz) OCM. Blue and red     differ if the latter is changed in either dc or ac mode. The
dots correspond to data taken with one OCM at a time. Bottom
                                                                    symbol hxjfi denotes the Fourier component of the signal x
plot: comparison between the lines resulting from fitting the
single-frequency case with Eq. (2) and those of Eqs. (6) and (8)
                                                                    at the frequency f, i.e., the projection of x on f. The signed
from the dual-frequency case. The two final offsets agree           amplitudes S are defined as:
roughly: 32  4 μm from the single-frequency excitation and                                                   ðzÞ
29  2 μm from the dual-plane modulation. The data shown here               Sfhxjf z ig ¼ jhxjf z ijsgnfcosðψ x − ψ z Þg;    ð11Þ
corresponds to the case BPM l ¼ 5.                                          ðzÞ
                                                                    where ψ x is the phase of the BPMs signal x at the steerer
                                                                    frequency f z corresponding to the z plane while ψ z is the
very well the linear fits obtained from the single-frequency        phase of that steerer signal. All projections and phases can
modulation. In this example, the offsets of the closest BPM         be obtained from the BPM readings of the FA. Explicit
to the quadrupole obtained from the two approaches are              formulas are given in Appendix A. There it is also shown
32  4 μm (single frequency) and 29  2 μm (dual fre-                                     ðzÞ
                                                                    how the difference ψ x − ψ z is either 0 or π. Thus, the sign
quency). These results are obtained after applying the same
                                                                    of S is either positive or negative and shall correspond to
statistics described at the end of Sec. II A.
                                                                    the one of the ORM coefficients.
                                                                       ℜfhxl j0ig represents thus the real part of the dc
    C. Dual-plane ac excitation and beam coupling                   component of the horizontal beam orbit during the simul-
   If betatron coupling, BPMs rolls or OCMs tilts cannot be         taneous horizontal and vertical ac OCM excitation.
neglected, Eq. (1) needs to be modified by including the            Sfhxl jf h ig is the signed amplitude of the horizontal
effective off-diagonal ORM terms Rxy , Ryx                          position at frequency f h and is the leading term along
                                                                    with the dc component. The term Sfhxl jf v ig accounts for
xk ðtÞ − x0k ¼ Rxx                   xy
                kj ðhj ðtÞ − h0 Þ þ Rkp ðvp ðtÞ − v0 Þ
                                                                    betatron coupling and is proportional to the horizontal
                                                                    beam response to the vertical OCM. Equivalent definitions
yn ðtÞ − y0n ¼ Ryy                   yx
                np ðvp ðtÞ − v0 Þ þ Rnj ðhj ðtÞ − h0 Þ:      ð9Þ    and considerations apply to the vertical data.

                                                              012802-5
MARTÍ, BENEDETTI, IRISO, and FRANCHI                                        PHYS. REV. ACCEL. BEAMS 23, 012802 (2020)

  The formulas in Eq. (10) are the core of the FBBA and                 In the case of ac excitation of the normal or skew
apply to any quadrupole type (normal or skew) and                     quadrupole, the coefficients D read
modulation (dc or ac). The coefficients M read
                       Dx Dyv − Dxv Dy   Y                                 Dx ¼ Sfhxk jf s ig
            Mh ¼ −                      ¼ hk
                       Dxh Dyv − Dxv Dyh X hk                              Dy ¼ Sfhyk jf s ig
                   Dxh Dy − Dx Dyh   Y                                    Dxh ¼ Sfhxk jf h þ f s ig þ Sfhxk jf h − f s ig
            Mv ¼ −                  ¼ vk :                   ð12Þ
                   Dxh Dyv − Dxv Dyh X vk
                                                                          Dyv ¼ Sfhyk jf v þ f s ig þ Sfhyk jf v − f s ig
For a dc (normal or skew) quadrupole scan, the coefficients               Dxv ¼ Sfhxk jf v þ f s ig þ Sfhxk jf v − f s ig
D become
                                                                          Dyh ¼ Sfhyk jf h þ f s ig þ Sfhyk jf h − f s ig:       ð14Þ
             Dx ¼ ℜfhxk2 j0ig − ℜfhxk1 j0ig
             Dy ¼ ℜfhyk2 j0ig − ℜfhyk1 j0ig                           An analytic derivation is given in Sec. A 3 of Appendix A.
                                                                      f s is the (known) skew quadrupole excitation frequency.
            Dxh ¼ Sfhxk2 jf h ig − Sfhxk1 jf h ig
                                                                         Throughout this paper, Mh;v are computed as the slopes
            Dyv ¼ Sfhyk2 jf v ig − Sfhyk1 jf v ig                     of the linear fits Y hk ¼ Mh X hk and Y vk ¼ Mv X vk , where
            Dxv ¼ Sfhxk2 jf h ig − Sfhxk1 jf h ig                     the denominator X and numerator Y are measured at all
                                                                      BPMs. An example of such fits in the case of a normal dc
            Dyh ¼ Sfhyk2 jf h ig − Sfhyk1 jf h ig:           ð13Þ     varied quadrupole for BPM l ¼ 5 is shown in Fig. 5. The
                                                                      vertical offset measured via Eq. (10), i.e., taking betatron
  An analytic derivation is given in Sec. A 2 of Appendix A.
                                                                      coupling into account, is 30.5  0.4 μm, compatible to
Labels 1 and 2 refer to data acquired at the quadrupole
                                                                      the value obtained from uncoupled analysis of Fig. 4
current I 0 − ΔI and at I 0 þ ΔI, respectively.




FIG. 5. Example of a Mh (top) and Mv (bottom) linear fit
using the combined data of the 120 BPMs before and after a dc         FIG. 6. Example of a Mh and Mv linear fit using the combined
normal quadrupole changed by 2.5A. Quadrupole and steerer             data of the 120 BPMs before and after a dc skew quadrupole
setting are the same of Fig. 4 i.e., BPM l ¼ 5. Using Eq. (10), the   changed by 2.5A. Using Eq. (10), the resulting offsets are x0l ¼
resulting vertical offset taking coupling effects into account        330.7  0.5 μm and y0l ¼ 265.3  7.0 μm. The data shown here
is y0l ¼ 30.5  0.4 μm.                                               corresponds to the case l ¼ 10.


                                                               012802-6
FAST BEAM-BASED ALIGNMENT …                                               PHYS. REV. ACCEL. BEAMS 23, 012802 (2020)

(29  2 μm) and the single-frequency excitation                     uncertainty. In particular one can check that a π=2 rotation,
(32  4 μm). For these specific quadrupole and BPM,                 that is x ⇒ y and y ⇒ −x which implies Dx ⇒ Dy ,
the third (coupling) term in the r.h.s of Eq. (10) which            Dy ⇒ −Dx , Dxh ⇒ Dyh , Dyh ⇒ −Dxh , Dxv ⇒ Dyv and
is proportional to Mh is relatively small, ≃0.1 μm. Larger          Dyv ⇒ −Dxv , leaves Eq. (12) unchanged.
deviations between this more general analysis and the one              A second example of linear fit to infer the coefficients M
(ignoring coupling) described in the previous section are to        and, thus, the offsets of a dc skew quadrupole next to the
be expected at quadrupoles with larger offsets (i.e., larger        BPM l ¼ 10 is shown in Fig. 6.
coefficients M) and/or larger coupled motion (i.e., large              A last example of analysis, this time performed with an
cross-term signed amplitudes Sfhxl jf v ig and Sfhyl jf h ig).      ac modulation of the same skew quadrupole of Fig. 6 is
    The uncertainty associated to the resulting BPM-to-quad         shown in Fig. 7. The horizontal offsets inferred with the dc
offsets is obtained by error propagation in Eq. (10). The           and ac excitation are 330.7  0.5 μm and 311.3  2.8 μm,
uncertainty associated to the M factors is obtained from the        respectively. The vertical offsets are also discordant
fit. The uncertainty associated to all projections hxjfi and        265.3  7.0 μm and 288.3  2.3 μm. In Sec. V it will be
hyjfi is obtained from the corresponding BPM data with              shown how a systematic discrepancy between the results
neither OCM nor quadrupole excitation. An example of                from a dc and an ac skew quadrupole variation affects all
such baseline noise is shown in Fig. 9.                             magnets at ALBA.
    In order to test the validity of Eq. (9) for large BPM rolls,
we reanalyzed the same data but rotating the x and y
readings of all the BPMs by different angles, except BPM l.          III. OPTIMIZING THE EXPERIMENTAL SETUP
Irrespective to the angle artificially applied the result              In practice, the experimental set-up has a certain param-
changes two orders of magnitude less that the assigned              eter settings that can influence the precision of our analysis.
                                                                    For this reason, we provide next a study of these param-
                                                                    eters, like the waveform amplitude and frequency or the
                                                                    BPM buffer length. The OCM waveform is defined by a
                                                                    series of discrete current set points separated by 80 μs and
                                                                    pre-loaded in the magnet power supply. The waveform is
                                                                    repeated continuously so that the current in the corrector
                                                                    coils emulates a sinusoidal profile. However, if the
                                                                    requested rate of change in the current is too large, the
                                                                    output current may not be able to follow the desired curve.
                                                                    For each waveform frequency there is thus an upper limit
                                                                    on the achievable effective sine amplitude. Above that
                                                                    maximum, the effective amplitude will no longer increase
                                                                    and the output current will exhibit spikes and disconti-
                                                                    nuities on top of the pure sinusoidal signal, introducing
                                                                    harmonics at higher frequencies.
                                                                       Figure 8 shows the maximum orbit distortion at the 120
                                                                    ALBA BPMs as a function of the OCM waveform
                                                                    amplitude and frequency. As in the standard BBA, a
                                                                    maximum orbit distortion of around 0.5 mm is created
                                                                    during the FBBA. According to Fig. 8 this is only possible
                                                                    at frequencies lower than 7 Hz. On the other hand, from 0 to
                                                                    15 Hz the BPMs noise decreases strongly as the frequency
                                                                    increases—see Fig. 9. An amplitude of 0.5 A and frequen-
                                                                    cies of 7 and 6 Hz for the horizontal and vertical plane,
                                                                    respectively, were found to be a reasonable trade-off for the
                                                                    quadrupole FBBA.
                                                                       ALBA skew quadrupole and OCM power supplies are
                                                                    identical and a similar approach was followed to determine
FIG. 7. Example of a Mh and Mv linear fit using the combined        the skew quadrupole frequency in ac mode. During the dc
data of the 120 BPMs with a skew quadrupole ac modulation (at       FBBA of normal and skew quadrupoles, their current is
1.6 Hz) of the same skew quadrupole of Fig. 6. By using Eq. (12)    varied by 2.5 A (the same change of the standard BBA at
and Eq. (14), the resulting offsets are x0l ¼ 311.3  2.8 μm and    ALBA). Following an extrapolation of the data in Fig. 8, in
y0l ¼ 288.3  2.3 μm. The data shown here corresponds to the        order to use the same current change during the ac skew
case l ¼ 10.                                                        quadrupole FBBA, its frequency is then limited to 1.6 Hz.

                                                              012802-7
MARTÍ, BENEDETTI, IRISO, and FRANCHI                                      PHYS. REV. ACCEL. BEAMS 23, 012802 (2020)

                                                                       The skew quadrupole trim coils are mounted on the
                                                                    sextupole magnets. When the former are powered the
                                                                    magnetic symmetry of the six poles is broken. Magnetic
                                                                    simulations show that the effect of the trim coils is sizable
                                                                    in displacing the magnetic center vertically (0.1 mm=A)
                                                                    but also quite linear. For this reason, FBBA of dc quadru-
                                                                    poles are performed both at −2.5 A and þ2.5 A. FBBA of
                                                                    ac quadrupole are expected to be less sensitive to this issue
                                                                    because of the continuous and symmetric variation of the
                                                                    trim coil current.
                                                                       The optimum acquisition time (i.e., the length of the FA
                                                                    buffer to be Fourier analyzed) was determined by evalu-
                                                                    ating the BPM-to-quad offset errors as a function of the
                                                                    used buffer length. To this end two different sources of error
                                                                    are considered. The first is the uncertainty in evaluating the
                                                                    coefficients Mh and Mv (group error). The second is the
                                                                    error in evaluating the BPM offsets via Eq. (10) (buffer
                                                                    error). Despite the names, both depend on the length of the
                                                                    FA buffer used for the analysis.
                                                                       For the normal quadrupole FBBA, a test was carried out
                                                                    on 5 BPMs with an acquisition time of 5.5 s and the
FIG. 8. Maximum orbit distortion over all the 120 BPMs as a
function of the waveform amplitude and frequency. A very            resulting offsets were taken as reference. These were then
similar behavior is observed in both planes.                        reevaluated after reducing the same FA buffer by steps of
                                                                    50 ms. Buffer errors between the reference offsets and the
                                                                    ones from the reduced buffer were then computed, and the
                                                                    resulting errors are averaged over the used BPMs subgroup.
                                                                    The group errors are recalculated for every different buffer
                                                                    and averaged for the 5 BPMs. Figure 10 presents an
                                                                    example of such error analysis. For acquisition times of
                                                                    around 1.5 s, both buffer and group errors are below 3 μm.




FIG. 9. Example of the 120 BPMs mean spectral noise at
18 mA (the typical beam current for a BBA measurement at
ALBA). The noise decreases rapidly from zero to 15 Hz. It is
worthwhile stressing the fact that while the BBA is influenced by
the integrated noise spectrum (1.5 μm and 0.8 μm rms in the two
planes, including the 50 Hz peak), the novel FBBA is affected       FIG. 10. Measured buffer and group offset errors, averaged
only by the noise at the selected frequencies (which is below       over 5 BPMs, as a function of the acquisition time for a (dc)
10−2 μm).                                                           normal quadrupole FBBA.


                                                              012802-8
FAST BEAM-BASED ALIGNMENT …                                            PHYS. REV. ACCEL. BEAMS 23, 012802 (2020)

                                                                 TABLE I. Summary of the optimized measurement parameters
                                                                 for which the expected buffer error in the evaluation of the offset
                                                                 is around 3 μm, and the group error of about 1 μm for the normal
                                                                 quadrupole case and 10 μm in the skew quadrupole case.

                                                                 OCM horizontal frequency fh                                7 Hz
                                                                 OCM vertical frequency f v                                 6 Hz
                                                                 ac skew quadrupole frequency fs                           1.6 Hz
                                                                 rms orbit distortion (H,V)                                0.5 mm
                                                                 ΔI OCM                                                     0.5 A
                                                                 ΔI normal quadrupole                                       2.5 A
                                                                 ΔI skew quadrupole                                         2.5 A
                                                                 Acquisition time (normal quadrupole)                       1.5 s
                                                                 Acquisition time (skew quadrupole)                         6.0 s


                                                                 with the buffer size. In that case, the group error is limited
                                                                 by a real discrepancy between BPMs in the Mh and Mv fit.
                                                                 The value at which the group error saturates indicates
                                                                 how well our linear model given by Eq. (9) corresponds to
                                                                 the reality.
                                                                    For the dc skew quadrupole FBBA the same error
FIG. 11. Measured average of 5 BPM offset errors as a function
                                                                 analysis was carried out, this time increasing the buffer
of the acquisition time for a (dc) skew quadrupole FBBA.         size to 6.5 s. The results are shown in Fig. 11. Buffer and
                                                                 group errors in the two planes are similar already after 2 or
                                                                 3 s, though at a level much higher than the one observed for
Longer acquisition times would further reduce systematic         normal quadrupoles (∼10 μm compared to less than 3 μm).
errors, though at slower pace and at the price of longer         This was to be expected, since skew quadrupoles have a
numerical calculations. An acquisition time of 1.5 s was         much weaker impact on the beam motion (and hence on the
found then to be a acceptable trade-off.                         BPM signal) compared to normal quadrupoles, where the
   Notice that in Fig. 10 the group error rapidly (above         buffer error reaches the 3 μm level around 6 s. The group
0.5 s) converges to a given value, and it does not decrease      error also is at the level of 3 μm in the horizontal plane.
                                                                    In the case of the ac skew quadrupole FBBA, the buffer
                                                                 and group errors as a function of the acquisition time are
                                                                 shown in Fig. 12. Again, at 6 s the buffer error reaches the
                                                                 3 μm level. At 6 s acquisition time the ac method has
                                                                 similar buffer errors but smaller group errors, specially in
                                                                 the vertical plane.
                                                                    Table I summarizes the optimized parameters of the
                                                                 experimental setup.
                                                                    The mechanical alignment of both quadrupole magnets
                                                                 and sextupole magnets during the ALBA storage ring
                                                                 installation is detailed in [13]. According to that, the
                                                                 alignment tolerances are in both cases around 30 μm.
                                                                 Also, both for quadrupoles and sextupoles, the coils
                                                                 mounted in the iron poles are cooled in parallel, which
                                                                 ensures the same equilibrium temperature and magnetic
                                                                 stated the poles and hence keeps the magnetic center.

                                                                 IV. ESTIMATION OF THE METHOD ACCURACY
                                                                    In the previous section the dependence of the measure-
                                                                 ment buffer and group errors on the acquisition time has
                                                                 been studied. There are other sources of uncertainty, for
                                                                 instance the beam orbit angle at the quadrupole to be
FIG. 12. Measured average of 5 BPM offset errors as a function   aligned (the lower, the better) and the distance between the
of the acquisition time for a ac skew quadrupole FBBA.           magnet and the nearest BPM (again, the lower, the better).

                                                           012802-9
MARTÍ, BENEDETTI, IRISO, and FRANCHI                                    PHYS. REV. ACCEL. BEAMS 23, 012802 (2020)

TABLE II. Standard deviation values for the Gaussian error        measurement BPM offsets is computed first, and averaged
distribution used to generate the 100 perturbed lattices.         over the 100 model lattices then, to generate a single
                                                                  number for each comparison. In general all methods have a
Girder (H, V) positioning error                        150 μm
Magnet (H, V) positioning error                         25 μm
                                                                  similar accuracy (lines 2–6), which is around 10% of the
Girder and magnet tilt error                           50 μrad    expected offsets (first line). Differences among the methods
Dipole and quadrupole field error                        0.1%     are lower: For normal quadrupole offsets the agreement is
                                                                  below 4 μm (lines 7–9), while for skew quadrupoles it is
                                                                  5 μm (last line). These error estimates shall be combined
The orbit angle is difficult to estimate experimentally,          with those estimated from the experimental parameters of
though readily accessible in simulations. In this section,        Sec. III.
we present results from numerical tests aiming at quantify-
ing the expected accuracy induced by these two factors.                     V. FBBA METHOD VALIDATION
Even though ac FBBA cannot be performed at ALBA
                                                                     In this section experimental results with the different
because the quadrupole power supplies do not have ac
                                                                  FBBA methods introduced in Sec. II are presented. In
capabilities, it is next included in the analysis for
                                                                  Sec. VA offsets measured via the standard quadrupole
completeness.
                                                                  BBA are compared to the ones obtained via the novel
   Standard BBA and novel FBBA (both ac and dc)
                                                                  FBBA, ignoring betatron coupling in the latter. In Sec. V B,
measurements have been simulated on a set of 100 lattice
models including realistic magnet and girder random               the impact of the corrections included in the FBBA to
displacements and tilts, magnetic field errors, as well as        account for coupling is evaluated on the same data sets.
an orbit distortion corrected at operational levels. The          Finally in Sec. V C, outcomes from the skew quadrupole
standard deviations of the (Gaussian) error distributions         FBBA via dc and ac excitation are examined.
are listed in Table II.                                              The ALBA storage ring comprises 32 combined-
   In order to minimize other sources of uncertainty in these     function-bending magnets, 112 quadrupoles, 32 skew
simulations, acquisition time and other experimental              quadrupoles, 88 OCMs per plane and 120 BPMs. Both
parameters of Table I have been artificially chosen so to         FBBA and BBA aim to align the 120 BPMs (or a subgroup
limit their contribution to the overall accuracy to 0.1 μm.       in the case of the skew quadrupoles) to their neighboring
For each one of the 100 lattice models and quadrupoles to         magnets. Even though each BPM has at least one quadru-
be aligned, the most suitable OCM was also chosen: this is        pole next to it, there are some quadrupole triplets with no
to take into account the different modulation of the lattice      BPM in between.
functions for each model and hence of the modified                   During the measurement, the ALBA storage ring fea-
effectiveness of the steerers.                                    tured a betatron coupling at the level of ϵy =ϵx ≃ 0.7%. The
   The accuracy achieved in simulations is summarized             experimental parameters of Table I were used.
in Table III. As figure of merit, the standard deviation of
the difference between the expected and the simulated                         A. Quadrupole BBA and FBBA
                                                                               (ignoring betatron coupling)
TABLE III. Top line: quadrupole rms offset inserted into the         The novel quadrupole FBBA is first compared to the
model and to be inferred. The corresponding BPM offsets are       standard BBA by ignoring betatron coupling in the harmonic
retrieved by the simulated measurements and the standard          analysis of the former, i.e., following the procedure described
deviation with respect to the expected value (model, lines 2–6)   in Sec. II A. The BPM-to-quad offsets of the 120 BPMs
or from another technique (BBA or dc FFBA, lines 7–10) is         obtained with the two methods are displayed in Fig. 13.
computed at all BPMs and averaged over 100 simulated lattices.    The overall agreement is rather good, although at some
                                           Horizontal Vertical    BPMs discrepancies are well beyond the estimated uncer-
                                                                  tainty. The standard deviation among all BPMs of the two
1   Model rms quadrupole offset             150 μm     150 μm     sets is around 15 μm in both planes. The expected buffer
    Mean difference between offsets:                              error originating from the finite measurement time is
2 (Normal quad.) BBA vs model                15 μm     12 μm      approximately 4 μm (see Sec. III) and the one stemming
3 (Normal quad.) dc FBBA vs model            16 μm     12 μm      from the lattice characteristics an additional 4 μm horizon-
4 (Normal quad.) ac FBBA vs model            16 μm     13 μm      tally and 2 μm vertically (see Table III). The observed
5 (Skew quad.) dc FBBA vs model              19 μm      9 μm      discrepancy is thus more than two times larger than expected.
6 (Skew quad.) ac FBBA vs model              19 μm      6 μm      The larger offsets measured around BPM no. 20 are
7 (Normal quad.) dc FBBA vs BBA               4 μm      2 μm      suspected to arise from important mechanical misalignment,
8 (Normal quad.) ac FBBA vs BBA               4 μm      3 μm      though to date this hypothesis could not be verified.
9 (Normal quad.) ac FBBA vs dc FBBA           0 μm      3 μm
                                                                     As far as the measurement reproducibility is concerned,
10 (Skew quad.) ac FBBA vs dc FBBA            0 μm      5 μm
                                                                  both BBA and FBBA of some quadrupoles were repeated

                                                           012802-10
FAST BEAM-BASED ALIGNMENT …                                                  PHYS. REV. ACCEL. BEAMS 23, 012802 (2020)




                                                                       FIG. 14. Reproducibility of both BBA and FBBA for the BPM
                                                                       no. 21 in the horizontal plane over 5 consecutive measurements
                                                                       with (curves 1) and without (curves 2) magnetic cycles.




FIG. 13. Standard BBA and FBBA measurements for the 120
ALBA BPMs. The upper plot shows the results for the horizontal
plane while the lower plot refers to the vertical plane. The results
exciting the OCMs in ac are shown in red while the dc case result
is shown in blue.


two consecutive times, and this repeated additional five
times after cycling all magnets and waiting ten minutes for
thermal stabilization. Results for the horizontal offset at the
BPM no. 21 are reported in Fig. 14, showing a global
reproducibility within the estimated error bar. Magnetic
hysteresis seems also to play no observable role, the curves
1 and 2 in the same plot (without magnet cycling) being
compatible with the five values measured after cycling the
quadrupoles. Similar observations were made at other
BPMs and in the vertical plane. Very similar level of
reproducibility were reported in [10].

       B. FBBA comparison with beam coupling
   A complete set of quadrupole FBBA is evaluated first by
ignoring betatron coupling (i.e., applying the procedure of
Sec. II B) and then by including its contribution (i.e.,
making use of the more general scheme of Sec. II C).                   FIG. 15. Difference between the FBBA offsets at the 120
The difference between the 120 BPM offsets inferred from               ALBA BPMs when including or excluding coupling effects (blue
the two approaches is displayed in the two plots of Fig. 15            curves, left axis). In order to attempt to correlate this difference
(left vertical scale). We observe a similar discrepancy in             with cross-talk between the two planes, the offsets in the opposite
both planes BPM offsets (6 μm rms and 7 μm rms in the                  plane are shown in the same plot (red curves, right axis).


                                                                012802-11
MARTÍ, BENEDETTI, IRISO, and FRANCHI                                      PHYS. REV. ACCEL. BEAMS 23, 012802 (2020)

horizontal and vertical plane respectively). Some of the                    C. Skew quadrupole FBBA comparison
largest differences in the horizontal plane appear to be               The FBBA analysis presented in Sec. II C was tested on
correlated to the largest vertical offsets (right vertical          32 skew quadrupoles (trim coils on sextupoles), both with
scale in the top plot of Fig. 15), similarly large horizontal       dc and ac excitation. Again, their offset are identified by the
BPM offsets induce large differences between the two                one inferred at the nearest BPM. The same optimized
approaches in the vertical plane (see bottom plot of the            experimental parameters of Table I were used. These are
same figure). This correlation, visible mainly around               expected to provide similar systematic errors (within
BPM no. 20, is to be expected since the coupling effect             4–5 μm), as shown in Sec. III and Table III. However,
is linear with the terms Mh and Mv [see Eq. (10)] so a              the use of an ac excitation reduces the measurement time
correlation of the coupling effect and the offset in the            by a factor 2.
opposite plane is expected. Instead, the large discrepancies           As shown in Fig. 16, the two methods yield very similar
observed in the vertical plane around BPM No. 60 and                results, especially in the horizontal plane. In the vertical
BPM No. 115 without any effect in the horizontal plane are          plane, the dc approach results in an almost systematically
not understood.                                                     larger offset (the average difference is 113 μm and the rms
   Also, this comparison provides an estimation of the              difference is 44 μm), which corresponds to a factor 20
FBBA and BBA accuracy lower limit for a given amount of             larger than the one expected from simulations (last row in
coupling (0.7% in our case), BPM rolls and OCM tilts                Table III).
(roughly 1% in our case): 6 μm and 7 μm rms in the two                 It is worth noticing that in the vertical plane, the dc
planes, respectively. This is smaller than the discrepancy          offsets have an average of 86 μm while the ac case the
between standard BBA and FBBA, which is at the level of             average is quite smaller - 26 μm. Regarding the method
15 μm rms (see Fig. 13).                                            precision, as in the example shown in Sec. II C, the ac case
                                                                    shows smaller error bars (when considering both planes).
                                                                       As in the case of the normal quadrupole FBBA, repeat-
                                                                    ability tests at some BPMs (i.e., skew quadrupoles) were
                                                                    performed. The results for BPM no. 21 (the same monitor
                                                                    as in Fig. 14) in the horizontal plane are displayed in
                                                                    Fig. 17. A weak dependence of the BPM offset on the
                                                                    magnetic hysteresis was observed for the dc FBBA during
                                                                    the first two sets of measurements only. That dependency




FIG. 16. FBBA measurements at the 32 BPMs nearest to skew
quadrupole magnets. The top plot shows the results for the          FIG. 17. Reproducibility of both dc and ac skew quadrupole
horizontal plane, while the bottom curves refer to the vertical     FBBA for the BPM no. 21 (6th skew quadrupole) in the
plane. Results obtained via the ac skew quadrupole excitation are   horizontal plane with (curves 1) and without (curves 2) magnetic
shown in blue while those from the dc scan are in red.              cycles.


                                                             012802-12
FAST BEAM-BASED ALIGNMENT …                                                  PHYS. REV. ACCEL. BEAMS 23, 012802 (2020)

on the magnetic history in the dc skew quadrupole case was            skew quadrupole magnets, the method could still gain a
discussed in Sec. III, it should be noticed that although it is       factor two in speed if the quadrupoles could be excited with
a week effect, the magnetic history effect brings the dc              an ac modulation.
measurement closer to the ac measurement. As in Fig. 14                  We also performed exhaustive simulations to evaluate
the error bars in Fig. 17 are smaller for the ac case even            the intrinsic accuracy of the method (in the order of 15 μm),
though they stem from a single measurement.                           and the expected differences between the standard and the
   It is worthwhile noticing how the measured skew                    FBBA. The experimental results show that both methods
quadrupole offsets of Fig. 16 are much larger than those              agree at the level of the intrinsic accuracy for the normal
of normal quadrupoles (see Fig. 13), reaching 500 μm at               quadrupole case. Larger and systematic deviations are
several locations, not only in sector 3 (region around BPM            observed in the vertical offsets of skew quadrupoles
no. 20 in Fig. 16) which is suspected to suffer from large            depending on their type of variation, dc or ac. The origin
mechanical misalignment. According to the mechanical                  of this discrepancy is still under investigation. Since the dc
alignment of sextupoles and quadrupoles discussed in                  skew quadrupole measurement is prone to hysteresis
Sec. III their difference should be in the range of 30 μm.            effects, we believe it is less accurate that than the ac
The origin of this large discrepancy is unknown and it is             version. Also the skew quadrupole ac measurement fea-
one of the crucial issues to tackle in future studies.                tures smaller error bars and is a factor 2 faster.
                                                                         It is our opinion that the larger scope of the FBBA,
                   VI. CONCLUSIONS                                    namely the inclusion of betatron coupling, BPM roll and
   We have developed a new method (called fast beam                   OCM tilt and the extension to skew quadrupoles (i.e., to
based alignment—FBBA) to align the beam through the                   sextupole alignment in most synchrotron light sources), and
center of the storage ring quadrupoles by using an ac                 its huge gain in rapidity allow to consider it as superior with
excitation of the OCMs and the 10 kHz BPM data                        respect to standard beam-based alignment techniques in
acquisition—a standard hardware in most synchrotron light             circular accelerators.
sources. The mathematical treatment presented in this paper
allows to perform dual plane BBA (horizontal and vertical),                              ACKNOWLEDGMENTS
includes betatron coupling, BPM roll and OCM tilt effects,
and extends this method to both normal and skew quadru-                  We would like to thank J. Marcos for the fruitful
pole magnets.                                                         discussions and his magnetic simulations, and E.
   With a careful choice of the experimental setup param-             Morales and J. Moldes for their contributions to the
eters, the technique speeds up the BBA at ALBA by about a             software and hardware development. We would also like
factor 30 (10 min vs 5 hours), and reaches the same level of          to extend our gratitude to the ALBA Operations and Power
precision. Furthermore and as shown for the case of the               Supply Groups for their help.


                              APPENDIX A: FBBA MATHEMATICAL BACKGROUND
   In this appendix assumptions and mathematics are detailed to derive the final formulas to infer the offset of (either normal
or tilted) quadrupoles from observable BPM data from the FA, assuming no knowledge of the ac steerers parameters
(strengths and possible tilts). Three hypotheses are needed in order to deduce these formulas: (1) For each quadrupole under
study, only one ac steerer per plane is excited at two different frequencies f h and f v , respectively. (2) The ac beam orbit
distortion at frequencies f h and f v is entirely due to the ac steerers, i.e., any other source of beam motion at those
frequencies must be negligible. (3) The amplitude of the ac steerers is also assumed to be invariant during the
quadrupole scan.

                                                       1. General formulas
  A more general version of Eq. (1) including betatron coupling reads

                                   xk ðtn Þ − x0k ¼ Rxx                      xy
                                                     kj ðhj ðtn Þ − h0j Þ þ Rkj ðvj ðtn Þ − v0j Þ;

                                   yk ðtn Þ − y0k ¼ Ryx                      yy
                                                     kj ðhj ðtn Þ − h0j Þ þ Rkj ðvj ðtn Þ − v0j Þ;                             ðA1Þ
                                                   yy
where tn is the FA sampling time, Rxx     kj and Rkj denote the on-diagonal ORM coefficients, while the off-diagonal ones
                                          xy      yx
(generated by betatron coupling) are Rkj and Rkj . As for Eq. (1), the index k denotes a generic BPM, whereas the label l will
later refer to the monitor closest to the quadrupole under study. The BPM reading corresponding to the centering of the
quadrupole will be ðx0l ; y0l Þ obtained with unknown (and not observable) OCM strengths ðh0j ; v0j Þ. Note that even though

                                                             012802-13
MARTÍ, BENEDETTI, IRISO, and FRANCHI                                                    PHYS. REV. ACCEL. BEAMS 23, 012802 (2020)

horizontal and vertical steerers share the same label j, they do not necessarily refer to the same magnet: In fact it may be
convenient make use of two different OCMs with different horizontal and vertical beta function in order to increase their
effectiveness. According to the hypotheses 1, and 2, the beam motion of interest for this study can be Fourier expanded as


                      X                                  X                                
                      2
                                                 ðmÞ
                                                           2
                                                                          ðmÞ                  1X 2
          xk ðtn Þ ¼       Xk;m cosð2πf m tn þ ψ xk Þ ¼ ℜ     ðXk;m e iψ xk
                                                                              Þe i2πf m tn
                                                                                             ¼       ðhxk jf m iei2πfm tn þ c:c:Þ;
                     m¼0                                  m¼0
                                                                                               2 m¼0
                     X2                                 X 2                                    X
                                                ðmÞ                      ðmÞ                   1 2
          yk ðtn Þ ¼     Y k;m cosð2πf m tn þ ψ yk Þ ¼ ℜ      ðY k;m eiψ yk Þei2πfm tn ¼             ðhyk jf m iei2πfm tn þ c:c:Þ;                  ðA2Þ
                     m¼0                                  m¼0
                                                                                               2 m¼0



where ℜ and c.c. represent the real part and the complex conjugate, respectively, whereas the hzjfi denotes the (observable)
projection of the BPM signal z onto the frequency f (a complex quantity, defined by amplitude and phase of the harmonic at
frequency f of the BPM reading, of which more in Appendix B). The above sums run over three indices, m ¼ 0 (dc mode,
f m ¼ 0), m ¼ 1 (f m ¼ f h ) and m ¼ 2 (f m ¼ f v ). The ac steerer functions can be also Fourier expanded. The hypotheses 1
and 2 allow to write them as


                                                                                          1
                        hj ðtn Þ ¼ ĥj cos ð2πf h tn þ ψ h Þ ¼ ℜfðĥj eiψ h Þei2πfh tn g ¼ ðhhj jf h iei2πfh tn þ c:c:Þ;
                                                                                          2
                                                                                          1
                        vj ðtn Þ ¼ v̂j cos ð2πf v tn þ ψ v Þ ¼ ℜfðv̂j e Þe
                                                                       iψ v i2πfv tn
                                                                                      g ¼ ðhvj jf v iei2πfv tn þ c:c:Þ:                             ðA3Þ
                                                                                          2


The amplitudes of the steerer excitation ĥj and v̂j are assumed to be not observable, as the archiving of their currents is
either absent or not reliable for a conversion into strengths. The above expressions are valid as long as the OCMs are
upright. A steerer tilted by an angle ω introduces a vertical (horizontal) motion component at frequency f h (f v ), namely


                                                1
                                      hj ðtn Þ ¼ ðCω;j hhj jf h iei2πfh tn þ S ω;j hvj jf v iei2πfv tn þ c:c:Þ;
                                                2
                                                1
                                      vj ðtn Þ ¼ ðCω;j hvj jf v iei2πfv tn þ S ω;j hhj jf h iei2πfh tn þ c:c:Þ:                                     ðA4Þ
                                                2

In the above definitions, Cω;j and S ω;j denote cos ωj and sin ωj , respectively. By inserting Eqs (A2) and (A4) in Eq. (A1),
the latter reads

                                                                                                                                                
1X 2
                                        0               1                                              xy 1
      ðhx jf ie   i2πf m t n þ c:c:Þ − xk ¼ Cω;j Rkj
                                                   xx     hh jf ie   i2πf h t n  þ c:c: − h0j þ Rkj           hv jf ie  i2πf v t n  þ c:c: − v0j
2 m¼0 k m                                               2 j h                                                2 j v
                                                                                                                                                   
                                                           1                                              xy 1
                                            þ S ω;j Rkj
                                                      xx     hv jf ie   i2πf    v t n þ c:c: − v0j þ Rkj         hh jf ie  i2πf    h t n þ c:c: − h0j ;
                                                           2 j v                                               2 j h
                                                                                                                                                
1X 2
                                                        1                                              yy 1
      ðhyk jf m iei2πfm tn þ c:c:Þ − y0k ¼ Cω;j Ryx       hh  jf  ie i2πf h tn þ c:c: − h
                                                                                              0j    þ R       hv  jf  iei2πfv tn þ c:c: − v
                                                                                                                                                 0j
2 m¼0                                              kj
                                                        2 j h                                          kj
                                                                                                             2 j v
                                                                                                                                                   
                                                           1                                              yy 1
                                            þ S ω;j Ryx      hv  jf  ie i2πf v tn
                                                                                      þ c:c: − v 0j   þ R        hh  jf  iei2πfh tn
                                                                                                                                         þ c:c: − h 0j   :
                                                      kj
                                                           2 j v                                          kj
                                                                                                               2 j h
                                                                                                                                                    ðA5Þ


Note that both sides in the above equations are real numbers, h0j and v0j being real quantities too. The above relations must
hold for each mode m ¼ 0, 1, 2 and at any time tn . They can be then split into two separate systems

                                                                      012802-14
FAST BEAM-BASED ALIGNMENT …                                                             PHYS. REV. ACCEL. BEAMS 23, 012802 (2020)

                              1
                                ðhx j0i þ c:c:Þ − x0k ¼ −Rxx               xy
                                                                kj h0j − Rkj v0j ;                                ðm ¼ 0Þ
                              2 k
                                                                                              
                           1                                       1
                             ðhxk jf h iei2πfh tn þ c:c:Þ ¼ Rxx      hh jf  ie i2πf h tn
                                                                                         þ c:c:  ;                ðm ¼ 1Þ
                           2                                 kj
                                                                   2 j h
                                                                                              
                           1                                 xy 1
                             ðhx jf ie   i2πfv tn
                                                  þ c:c:Þ ¼ Rkj      hv jf ie i2πfv tn
                                                                                         þ c:c: ;                 ðm ¼ 2Þ;                     ðA6Þ
                           2 k v                                   2 j v
                              1
                                ðhy j0i þ c:c:Þ − y0k ¼ −Ryx            yy
                                                             kj h0j − Rkj v0j ;                                   ðm ¼ 0Þ
                              2 k
                                                                                         
                           1                              yx    1
                             ðhy jf ie h n þ c:c:Þ ¼ Rkj
                                      i2πf  t                     hh jf ie h n þ c:c: ;
                                                                           i2πf  t                                ðm ¼ 1Þ
                           2 k h                                2 j h
                                                                                         
                           1                              yy 1
                             ðhy jf iei2πfv tn
                                               þ c:c:Þ ¼ Rkj      hv jf ie i2πfv tn
                                                                                    þ c:c: ;                      ðm ¼ 2Þ:                     ðA7Þ
                           2 k v                                2 j v
In the above expressions the original ORM coefficients have been replaced by effective ones, accounting for the effects of
tilted steerers:
                                                             xy
                                      kj ↔ Cω;j Rkj þ S ω;j Rkj
                                     Rxx         xx
                                                                                Rxy         xy
                                                                                 kj ↔ Cω;j Rkj þ S ω;j Rkj
                                                                                                        xx
                                                                                                                                               ðA8Þ
                                     Ryy         yy          yx
                                      kj ↔ Cω;j Rkj þ S ω;j Rkj ;               Ryx         yx          yy
                                                                                 kj ↔ Cω;j Rkj þ S ω;j Rkj .

Since the ORM coefficients are not observable through this technique and they will not be used in the final formulas, this
replacement and thus the steerer tilts do not impact the following derivation and analysis. The dc equations (m ¼ 0) can be
written as
                     ℜfhxk j0ig − x0k ¼ −Rxx        xy
                                          kj h0j − Rkj v0j                           x0k ¼ ℜfhxk j0ig þ Rxx        xy
                                                                                                         kj h0j þ Rkj v0j
                                                                              ⇒                                                                ðA9Þ
                     ℜfhyk j0ig − y0k ¼ −Ryx        yy
                                          kj h0j − Rkj v0j ;                         y0k ¼ ℜfhyk j0ig þ Ryx        yy
                                                                                                         kj h0j þ Rkj v0j .

The above equations are the starting point for the evaluation of the quadrupole offset ðx0l ; y0l Þ. The next step is to manipulate
their right-hand side (r.h.s.) in order to make only observable quantities appear. A physics consideration imposes some
constraints on the last two equations of both systems in Eqs. (A6)–(A8). Indeed, the ORM coefficients must be real numbers
and time-independent. Those equations must then hold simultaneously for both harmonics oscillating at f h and f v ,
resulting in
                                        hxk jf h i ¼ Rxx
                                                      kj hhj jf h i             hyk jf h i ¼ Ryx
                                                                                              kj hhj jf h i
                                        hxk jf h i ¼ Rxx
                                                       kj hhj jf h i
                                                                     
                                                                                hyk jf h i ¼ Ryx
                                                                                               kj hhj jf h i
                                                                                                            
                                                                                                                                              ðA10Þ
                                        hxk jf v i ¼ Rxy
                                                      kj hvj jf v i             hyk jf v i ¼ Ryy
                                                                                              kj hvj jf v i
                                        hxk jf v i ¼ Rxy           
                                                       kj hvj jf v i ;          hyk jf v i ¼ Ryy           
                                                                                               kj hvj jf v i .

The (real) ORM coefficients can be written as
                                                                                        
                                                          iθðRkj Þ
                                                                                            0;   if Rkj > 0
                                          Rkj ¼ jRkj je              ;     θðRkj Þ ¼                          :                               ðA11Þ
                                                                                            π; if Rkj < 0
The equations in the systems of Eq. (A10) can then be written as
                               ðhÞ                                                                ðhÞ                             yx
                                                        i½θðRkj Þþψ h                                                  i½θðRkj Þþψ h 
              jhxk jf h ijeiψ xk ¼ jRxx                                        jhyk jf h ijeiψ yk ¼ jRyx
                                                                xx
                                      kj jjhhj jf h ije                                                 kj jjhhj jf h ije
                              ðvÞ                               xy                                ðvÞ                             yy
                                                                                                                                              ðA12Þ
                                                       i½θðRkj Þþψ v ;                                                 i½θðRkj Þþψ v 
              jhxk jf v ijeiψ xk ¼ jRxy
                                      kj jjhvj jf v ije                         jhyk jf v ijeiψ yk ¼ jRyy
                                                                                                        kj jjhvj jf v ije                 .
                                                                                                                                               ðh;vÞ
For the above equations to be simultaneously valid (i.e., irrespective of the sign in the exponential term) BPM phases ψ x;y
and steerer phases ψ h;v must satisfy the following relations

                                               ðhÞ                                ðhÞ
                                             ψ xk − ψ h ¼ θðRxx
                                                             kj Þ               ψ yk − ψ h ¼ θðRyx
                                                                                                kj Þ
                                                                                                                                              ðA13Þ
                                               ðvÞ                                ðvÞ
                                             ψ xk − ψ v ¼ θðRxy
                                                             kj Þ;              ψ yk − ψ v ¼ θðRyy
                                                                                                kj Þ.



                                                                         012802-15
MARTÍ, BENEDETTI, IRISO, and FRANCHI                                         PHYS. REV. ACCEL. BEAMS 23, 012802 (2020)

The sign of any ORM coefficient can be then replaced by the difference between BPM and steerer phase. By combining
Eqs (A11)–(A13) the following expressions for the ORM element can be derived

                                                  Sfhxk jf h ig                 Sfhyk jf h ig
                                           kj ¼
                                          Rxx                           Ryx
                                                                         kj ¼
                                                      ĥj                           ĥj
                                                                                                                             ðA14Þ
                                               Sfhxk jf v ig                  Sfhyk jf v ig
                                         Rxy
                                          kj ¼               ;          Ryy
                                                                         kj ¼               ;
                                                   v̂j                            v̂j

where ĥj ¼ jhhj jf h ij and v̂j ¼ jhvj jf v ij are the steerer amplitudes introduced in Eq. (A4) and the signed amplitudes S are
defined as
                                                   ðhÞ                                                        ðhÞ
              Sfhxk jf h ig ¼ jhxk jf h ijsgnfcosðψ xk − ψ h Þg         Sfhyk jf h ig ¼ jhyk jf h ijsgnfcosðψ yk − ψ h Þg
                                                                                                                             ðA15Þ
                                                   ðvÞ                                                        ðvÞ
             Sfhxk jf v ig ¼ jhxk jf v ijsgnfcosðψ xk − ψ v Þg;         Sfhyk jf v ig ¼ jhyk jf v ijsgnfcosðψ yk − ψ v Þg.

Since the argument within the cosine terms are either 0 or π, the sign of S is either positive or negative. While BPM amplitudes
                             ðh;vÞ
jhx; yjf h;v ij and phases ψ x;y are direct observables from the harmonic analysis of FA data, the steerer phases ψ h;v are not.
Nevertheless they can be inferred from the BPM phases by making use of Eqs. (A11) and (A13), since
                                                                                                     1         ðhÞ
                                                                                                ψh ¼   mod ð2ψ xk ; 2πÞ
                ðhÞ
              ψ xk ¼ n1 π þ ψ h
                                                     ðhÞ
                                                  2ψ xk ¼ 2n1 π þ 2ψ h                               2
                                                                                                     1         ðvÞ
                ðvÞ
              ψ xk ¼ n2 π þ ψ v
                                                     ðvÞ
                                                  2ψ xk ¼ 2n2 π þ 2ψ v                          ψ v ¼ mod ð2ψ xk ; 2πÞ
                                        ⇒                                          ⇒                 2                       ðA16Þ
                ðhÞ
              ψ yk ¼ n3 π þ ψ h
                                                     ðhÞ
                                                  2ψ yk ¼ 2n3 π þ 2ψ h                               1         ðhÞ
                                                                                                ψ h ¼ mod ð2ψ yk ; 2πÞ
                ðvÞ                                  ðvÞ                                             2
              ψ yk ¼ n4 π þ ψ v                   2ψ yk ¼ 2n4 π þ 2ψ v ;                             1         ðvÞ
                                                                                                ψ v ¼ mod ð2ψ yk ; 2πÞ;
                                                                                                     2
where n1;2;3 are either 0 or 1 according to the sign of the corresponding ORM coefficient. The last system in the above equation
indicates that the steerer phase can be retrieved from the BPM reading in both planes and that it must be the same at all BPMs.
This allows to average over the two planes and among all N BPMs, namely

                                   with large coupling∶                               with ultra − low coupling∶
                         1 X
                           N
                                            ðhÞ                   ðhÞ                      1 X N
                                                                                                           ðhÞ
                 ψh ¼             ½mod ð2ψ xk ; 2πÞ þ mod ð2ψ yk ; 2πÞ             ψh ¼           mod ð2ψ xk ; 2πÞ
                        4N k¼1                                                            2N k¼1                             ðA17Þ
                         1 XN
                                        ðvÞ                ðvÞ                             1 XN
                                                                                                         ðvÞ
                 ψv ¼          ½mod ð2ψ yk ; 2πÞ þ mod ð2ψ xk ; 2πÞ;              ψv ¼          mod ð2ψ yk ; 2πÞ;
                        4N k¼1                                                            2N k¼1

The steerer phase ψ h;v is hence measurable and so are the signed amplitudes S of Eq. (A15). The choice of using either
the first or the second set of equations above may depend on the amount of coupling in the machine. Indeed the phases
  ðhÞ        ðvÞ
ψ yk and ψ xk originate from the BPM signal in the plane orthogonal to the excitation: With ultralow coupling the
measurement of that weak signal may be corrupted by noise and other sources, hence reducing accuracy and precision
in evaluating the steerer phases.
   Once the steerer phases are computed, the first system of Eq. (A16) can be used to asses the quality of the data: The
smaller the deviation from nπ of the difference between BPM and steerer phase, the more robust is the analysis.
   With both S and ψ h;v computed from observable BPM data, we can go back to Eq. (A9), substitute the ORM coefficients
with Eq. (A14) and evaluate the latter for the BPM l closest to the quadrupole under study:

                                                                                                        h0j
                                                                                                 Mh ¼
                           x0l ¼ ℜfhxl j0ig þ Sfhxl jf h igMh þ Sfhxl jf v igMv                       ĥj
                                                                                                                             ðA18Þ
                           y0l ¼ ℜfhyl j0ig þ Sfhyl jf v igMv þ Sfhyl jf h igMh ;                     v0j
                                                                                                 Mv ¼     ;
                                                                                                      v̂j


                                                            012802-16
FAST BEAM-BASED ALIGNMENT …                                                PHYS. REV. ACCEL. BEAMS 23, 012802 (2020)

where the coefficients M are observable and depend on the magnet under study and on the type of its variation (either dc or
ac). The formulas in Eq. (A18) are the core of the FBBA. In the next sections, explicit expressions for the coefficients M
will be derived for normal and skew quadrupoles, both with dc and ac excitation.
   It is worthwhile noticing that the second terms in the r.h.s. of Eq. (A18) scale with Sfhxl jf h ig and Sfhyl jf v ig,
respectively, i.e., with the beam response to the excitation in the corresponding plane: These are the leading terms along
with the dc beam responses ℜfhxl j0ig and ℜfhyl j0ig. The last terms scale instead with Sfhxl jf v ig and Sfhyl jf h ig,
respectively, i.e., with the beam response in the plane orthogonal to the steerer excitation: This in turn scales with (and
account for) betatron coupling in the machine and could be ignored in machines operating at ultralow coupling, where
jhxl jf v ij=jhxl jf h ij; jhyl jf h ij=jhyl jf v ij ≈ 0.1%. In order to have the same measurement sensitivity in both planes, OCMs
shall be chosen so to have jMh j ≃ jMv j.


                                           2. Coefficients M for a dc quadrupole
   In this section we derive the coefficients M for the case of a quadrupole, either normal, skew or partially rotated, whose
strength is varied at least one time and the BPM modes measured each time. By doing so, the dc modes of Eq. (A9) during
the quadrupole scan read


                     x0k ¼ ℜfhxk1 j0ig þ Rxx         xy
                                          kj1 h0j þ Rkj1 v0j         y0k ¼ ℜfhyk1 j0ig þ Ryx         yy
                                                                                          kj1 h0j þ Rkj1 v0j
                                                                                                                              ðA19Þ
                     x0k ¼ ℜfhxk2 j0ig þ Rxx         xy
                                          kj2 h0j þ Rkj2 v0j ;       y0k ¼ ℜfhyk2 j0ig þ Ryx         yy
                                                                                          kj2 h0j þ Rkj2 v0j ;



where the labels 1 and 2 refer to the quadrupole current I 0 − ΔI and to I 0 þ ΔI, respectively. While varying the quadrupole
strength, neither the BPM readings centering the magnet (x0k , y0k ) nor the dc steerer strengths (ĥj , v̂j ) changes (hypothesis n.
3). The difference between the two equations in the above two systems then reads

                                                                                xy     xy
                              ℜfhxk2 j0ig − ℜfhxk1 j0ig ¼ ðRxx
                                                            kj1 − Rkj2 Þh0j þ ðRkj1 − Rkj2 Þv0j
                                                                   xx


                              ℜfhyk2 j0ig − ℜfhyk1 j0ig ¼ ðRyy     yy           yx     yx
                                                            kj1 − Rkj2 Þv0j þ ðRkj1 − Rkj2 Þh0j :                             ðA20Þ


None of the quantities in the r.h.s. of the above equations is actually observable. Nevertheless, by diving h0j by ĥj and v0j by
v̂j and applying Eqs. (A14) and (A18), the rhs are rewritten in terms of the coefficients M and of the observable signed
amplitudes S, namely


           ℜfhxk2 j0ig − ℜfhxk1 j0ig ¼ ðSfhxk1 jf h ig − Sfhxk2 jf h igÞMh þ ðSfhxk1 jf v ig − Sfhxk2 jf v igÞMv
           ℜfhyk2 j0ig − ℜfhyk1 j0ig ¼ ðSfhyk1 jf h ig − Sfhyk2 jf h igÞMh þ ðSfhyk1 jf v ig − Sfhyk2 jf v igÞMv :            ðA21Þ


   By defining the following observable parameters, all computed from the harmonic analysis of BPM data before and
after varying the quadrupole strength


                                               Dx ¼ ℜfhxk2 j0ig − ℜfhxk1 j0ig
                                               Dy ¼ ℜfhyk2 j0ig − ℜfhyk1 j0ig
                                              Dxh ¼ Sfhxk2 jf h ig − Sfhxk1 jf h ig
                                              Dyv ¼ Sfhyk2 jf v ig − Sfhyk1 jf v ig
                                              Dxv ¼ Sfhxk2 jf h ig − Sfhxk1 jf h ig
                                              Dyh ¼ Sfhyk2 jf h ig − Sfhyk1 jf h ig;                                          ðA22Þ


Cramer’s rule can be used to invert the system of Eq. (A21), yielding

                                                            012802-17
MARTÍ, BENEDETTI, IRISO, and FRANCHI                                                PHYS. REV. ACCEL. BEAMS 23, 012802 (2020)

                                                               Dx Dyv − Dxv Dy   Y
                                                  Mh ¼ −                        ¼ hk
                                                               Dxh Dyv − Dxv Dyh X hk
                                                               Dxh Dy − Dx Dyh   Y
                                                   Mv ¼ −                       ¼ vk :                                                    ðA23Þ
                                                               Dxh Dyv − Dxv Dyh X vk

   In principle M can be evaluated at every BPM k and averaged over all monitors to obtain a more precise value.
However, the denominator in Eq. (A23) can be very small, hence risking to spoil the final result. To overcome this
difficulty it was found to be more convenient to evaluate the above denominator X and numerator Y at all BPMs and to
infer the coefficients M from the slope of their linear fit Y hk ¼ Mh X hk and Y vk ¼ Mv X vk .

                                               3. Coefficients M for an ac quadrupole
   In this section we derive the coefficients M for the case of a quadrupole, either normal, skew or partially rotated, whose
strength is varied continuously via an ac (harmonic) excitation during the modulation of the OCMs. This reduces the
measurement time by a factor two, since the measurement does not need to be repeated for two (dc) values of the quadrupole
strengths. However, this approach requires a more complex mathematical derivation, since the starting point of Eq. (A1)
can no longer be used and a more general formalism needs to be deployed. In presence of an ac quadrupole modulation
Eq. (A1) reads

                        xk ðtn Þ − x0k ¼ ðRxx                                   xy    xy
                                           kj þ ρkj ðtn ÞÞðhj ðtn Þ − h0j Þ þ ðRkj þ ρkj ðtn ÞÞðvj ðtn Þ − v0j Þ;
                                                 xx


                        yk ðtn Þ − y0k ¼ ðRyy    yy                             yx    yx
                                           kj þ ρkj ðtn ÞÞðvj ðtn Þ − v0j Þ þ ðRkj þ ρkj ðtn ÞÞðhj ðtn Þ − h0j Þ;                         ðA24Þ

where ρðtn Þ ¼ ρ̂ cos ð2πf s tn þ ψ s Þ is the ac variation of the ORM coefficients induced by the quadrupole excitation at
(known) frequency f s and where ρðtn Þ ¼ ρ̂ cos ð2πf s tn þ ψ s Þ is the ac variation of the ORM coefficients induced by the
quadrupole excitation at (known) frequency f s and phase ψ s (unknown a priori but measurable from BPM data as shown
later). As for the dc case, both on-diagonal ORM blocks (ρxx , ρyy ) and off-diagonal (ρxy , ρxy ) are let varying in order to
account for any possible quadrupole rotation. The terms that do not include ρðtn Þ in the above r.h.s. are the same of the dc
case studied so far and excite three modes: m ¼ 0 (f m ¼ 0), m ¼ 1 (f m ¼ f h ) and m ¼ 2 (f m ¼ f v ). The new ac terms
generate five additional modes in the recorded orbit. The horizontal ac quadrupole terms can indeed be rewritten as
                                                                                                                                 
                                                            xx iðmkj πþψ s Þ i2πfs tn         1
           ρ̂xx
             kj cosð2πf s tn þ ψ s Þðhj ðtn Þ − h0j Þ ¼ ℜfρ̂kj e            e         g         ðhh jf iei2πfh tn þ c:c:Þ − h0j
                                                                                              2 j h
                                                        1
                                                      ¼ ðhρxx   jf iei2πfs tn þ c:c:Þðhhj jf h iei2πfh tn − h0j þ c:c:Þ
                                                        4 kj s
                                                        1                       i2πðfh þfs Þtn  1
                                                      ¼ hρxxkj jf s ihhj jf h ie               þ hρxx   jf i hhj jf h iei2πðfh −fs Þtn
                                                        4                                       4 kj s
                                                          h0j xx
                                                        −      hρ jf iei2πfs tn þ c:c:;
                                                           2 kj s
                                                                                                                               
             xy                                              xy iðnkj πþψ s Þ i2πf s tn   1
           ρ̂kj cosð2πf s tn þ ψ s Þðvj ðtn Þ − v0j Þ ¼ ℜfρ̂kj e              e         g ðhvj jf v ie i2πf v tn
                                                                                                                 þ c:c:Þ − v0j
                                                                                          2
                                                        1
                                                      ¼ ðhρxy   jf iei2πfs tn þ c:c:Þðhvj jf v iei2πfv tn − v0j þ c:c:Þ
                                                        4 kj s
                                                        1                                       1
                                                      ¼ hρxy   jf ihv jf iei2πðfv þfs Þtn þ hρxy        jf i hvj jf v iei2πðfv −fs Þtn
                                                        4 kj s j v                              4 kj s
                                                          v0j xy
                                                        −     hρ jf iei2πfs tn þ c:c:;                                                    ðA25Þ
                                                           2 kj s

where c.c. denotes as usual the complex conjugate, hρxx               xx iðmkj πþψ s Þ , hρxy jf i ¼ ρ̂xy eiðnkj πþψ s Þ . The amplitude of
                                                        kj jf s i ¼ ρ̂kj e                 kj s        kj
the ORM modulation ρ̂kj depends on the maximum ac modulation imparted by the quadrupole and is always positive.
In order to account for a change in the sign of the ac ORM coefficients along the ring, which depend on the BPM k and
steerer j, the integer numbers mkj and nkj are introduced in the corresponding phase terms. The five additional
horizontal orbit modes oscillate then at the frequencies ðf h  f s Þ, ðf v  f s Þ and f s . The ac OCM terms hj ðtn Þ and
vj ðtn Þ is expanded as in Eq. (A4), whereas the BPM reading xk ðtn Þ is the same of Eq (A2) with the sum now running

                                                                   012802-18
FAST BEAM-BASED ALIGNMENT …                                                                 PHYS. REV. ACCEL. BEAMS 23, 012802 (2020)

over eight indices m ¼ 0; 1; …; 7. The first three are the same of Eq. (A6), whereas the additional five stem from
Eq. (A25) and read

                 1                                           1
                   ðhx jf þ f s iei2πðfh þfs Þtn þ c:c:Þ ¼ hρxx       jf ihv jf iei2πðfh þfs Þtn þ c:c:;                      ðm ¼ 3Þ
                 2 k h                                       4 kj s j h
                 1                                           1
                   ðhxk jf h − f s iei2πðfh −fs Þtn þ c:c:Þ ¼ hρxx   jf i hhj jf h iei2πðfh −fs Þtn þ c:c:;                  ðm ¼ 4Þ
                 2                                           4 kj s
                 1                                           1
                   ðhxk jf v þ f s iei2πðfv þfs Þtn þ c:c:Þ ¼ hρxy    jf ihv jf iei2πðfv þfs Þtn þ c:c:;                      ðm ¼ 5Þ
                 2                                           4 kj s j v
                 1                                           1
                   ðhx jf − f s iei2πðfv −fs Þtn þ c:c:Þ ¼ hρxy      jf i hvj jf v iei2πðfv −fs Þtn þ c:c:;                  ðm ¼ 6Þ
                 2 k v                                       4 kj s
                 1                                    1                         xy
                   ðhxk jf s iei2πfs tn þ c:c:Þ ¼ − ðh0j hρxxkj jf s i þ v0j hρkj jf s iÞe
                                                                                          i2πfs tn
                                                                                                    þ c:c:;                  ðm ¼ 7Þ;   ðA26Þ
                 2                                    2

Once again, the above relations must hold at any time. This implies that all factors in front of the phasors ei2πftn of both
hand sides must be equal. The following systems of equations are then obtained

                   hxk jf h þ f s i ¼ 12 hρxx
                                           kj jf s ihhj jf h i                     hxk jf h þ f s i ¼ 12 hρxx       
                                                                                                            kj jf s i hhj jf h i
                                                                                                                                

                   hxk jf h − f s i ¼ 12 hρxx       
                                           kj jf s i hhj jf h i                    hxk jf h − f s i ¼ 12 hρxx
                                                                                                            kj jf s ihhj jf h i
                                                                                                                                


                   hxk jf v þ f s i ¼ 12 hρxy
                                           kj jf s ihvj jf v i                     hxk jf v þ f s i ¼ 12 hρxy       
                                                                                                            kj jf s i hvj jf v i
                                                                                                                                


                   hxk jf v − f s i ¼ 12 hρxy       
                                           kj jf s i hvj jf v i                    hxk jf v − f s i ¼ 12 hρxy
                                                                                                            kj jf s ihvj jf v i
                                                                                                                               

                   hxk jf s i ¼ −v0j hρxy
                                       kj jf s i − h0j hρkj jf s i;
                                                         xx
                                                                                   hxk jf s i ¼ −v0j hρxy                        
                                                                                                        kj jf s i − h0j hρkj jf s i ;
                                                                                                                          xx



which, after some algebra, can be compacted as

                                                                  ðhþsÞ        1
                                       jhxk jf h þ f s ijeiðψ xk         −ψ s −ψ h Þ
                                                                              ¼ ρ̂xx ĥ eimkj π
                                                                               2 kj j
                                                               ðh−sÞ           1
                                       jhxk jf h − f s ijeiðψ xk þψ s −ψ h Þ ¼ ρ̂xx ĥ e∓imkj π
                                                                               2 kj j
                                                               ðvþsÞ           1
                                       jhxk jf v þ f s ijeiðψ xk −ψ s −ψ v Þ ¼ ρ̂xy v̂ einkj π
                                                                               2 kj j
                                                               ðv−sÞ           1
                                       jhxk jf v − f s ijeiðψ xk þψ s −ψ v Þ ¼ ρ̂xy v̂ e∓inkj π
                                                                               2 kj j
                                                         ðsÞ
                                       jhxk jf s ijeiðψ xk −ψ s Þ ¼ −v0j ρ̂xy
                                                                            kj e
                                                                                 inkj π
                                                                                         − h0j ρ̂xx
                                                                                                 kj e
                                                                                                     imkj π
                                                                                                             :                          ðA27Þ

 ðhsÞ   ðvsÞ        ðsÞ
ψ xk , ψ xk and ψ xk denote the observable phases of the new BPM modes, whereas ψ h and ψ v are the steerer phases
measurable from the BPM data via Eq. (A17). The phase of the ac quadrupole modulation ψ s is still unknown, though
formulas to infer it from BPM data will be given at the end of this section. Once again, for the above equations to be
simultaneously valid (i.e., irrespective of the sign in the exponential term) the following relations between the different
phases must hold

                                                         ðhþsÞ
                                                       ψ xk       − ψ s − ψ h ¼ mkj π
                                                         ðh−sÞ
                                                       ψ xk þ ψ s − ψ h ¼ −mkj π
                                                         ðvþsÞ
                                                       ψ xk       − ψ s − ψ v ¼ nkj π
                                                         ðv−sÞ
                                                       ψ xk       þ ψ s − ψ v ¼ −nkj π
                                                                     ðsÞ
                                                                   ψ xk − ψ s ¼ okj π;                                                  ðA28Þ


                                                                     012802-19
MARTÍ, BENEDETTI, IRISO, and FRANCHI                                            PHYS. REV. ACCEL. BEAMS 23, 012802 (2020)

which make both sides in all five equations of the system in Eq. (A27) be real quantities. The (unknown) integer okj is
introduced in the last equation in order to account for the sign of its r.h.s.. The signed amplitude S can be again
introduced as


                                                                                      ðhþsÞ
                                 Sfhxk jf h þf s ig ¼ jhxk jf h þf s ijsgnfcosðψ xk           −ψ s −ψ h Þg
                                                                                 ðh−sÞ
                                 Sfhxk jf h −f s ig ¼ jhxk jf h −f s ijsgnfcosðψ xk þψ s −ψ h Þg
                                                                                      ðvþsÞ
                                 Sfhxk jf v þf s ig ¼ jhxk jf v þf s ijsgnfcosðψ xk           −ψ s −ψ v Þg
                                                                                 ðv−sÞ
                                 Sfhxk jf v −f s ig ¼ jhxk jf v −f s ijsgnfcosðψ xk þψ s −ψ v Þg
                                                                               ðsÞ
                                      Sfhxk jf s ig ¼ jhxk jf s ijsgnfcosðψ xk −ψ s Þg;                                         ðA29Þ


With the above definition, the system of Eq. (A27) can be rewritten as


                                                            1
                                       Sfhxk jf h þ f s ig ¼ ρ̂xx ĥ eimkj π
                                                            2 kj j
                                                            1
                                       Sfhxk jf h − f s ig ¼ ρ̂xx ĥ eimkj π
                                                            2 kj j
                                                            1
                                       Sfhxk jf v þ f s ig ¼ ρ̂xy v̂ einkj π
                                                            2 kj j
                                                            1
                                       Sfhxk jf v − f s ig ¼ ρ̂xy v̂ einkj π
                                                            2 kj j
                                                                         inkj π              imkj π
                                            Sfhxk jf s ig ¼ −v0j ρ̂xy
                                                                    kj e        − h0j ρ̂xx
                                                                                        kj e        ;                           ðA30Þ


where we have removed the  sings since eimkj π ¼ e−imkj π . Notice that the projections at ðf h  f s Þ and ðf v  f s Þ
generate the same r.h.s in Eq. (A30), but we use them both in order to improve the overall signal to noise ratio. The
sum between the equations in the system of Eq. (A27) reads


                                                                         v0j                                              h0j
          Sfhxk jf s ig ¼ −ðSfhxk jf v þ f s ig þ Sfhxk jf v − f s igÞ       − ðSfhxk jf h þ f s ig þ Sfhxk jf h − f s igÞ :    ðA31Þ
                                                                         v̂j                                              ĥj


An equivalent analysis of the vertical signal of Eq. (A24) leads to


                                                                         v0j                                              h0j
          Sfhyk jf s ig ¼ −ðSfhyk jf v þ f s ig þ Sfhyk jf v − f s igÞ       − ðSfhyk jf h þ f s ig þ Sfhyk jf h − f s igÞ :    ðA32Þ
                                                                         v̂j                                              ĥj


Equations (A31) and (A32) can be cast in a linear system equivalent to the one of the dc case, where the unknown
coefficients Mh ¼ h0j =ĥj and Mv ¼ v0j =v̂j appear:


                                                   −Dx ¼ Dxh Mh þ Dxv Mv
                                                   −Dy ¼ Dyh Mh þ Dyv Mv :                                                      ðA33Þ


where the auxiliary observable terms D read

                                                               012802-20
FAST BEAM-BASED ALIGNMENT …                                                PHYS. REV. ACCEL. BEAMS 23, 012802 (2020)

                                        Dx ¼ Sfhxk jf s ig
                                        Dy ¼ Sfhyk jf s ig
                                       Dxh ¼ Sfhxk jf h þ f s ig þ Sfhxk jf h − f s ig
                                       Dyv ¼ Sfhyk jf v þ f s ig þ Sfhyk jf v − f s ig
                                       Dxv ¼ Sfhxk jf v þ f s ig þ Sfhxk jf v − f s ig
                                       Dyh ¼ Sfhyk jf h þ f s ig þ Sfhyk jf h − f s ig:                                 ðA34Þ

  Once again, Cramer’s rule can be used to invert the system of Eq. (A33), yielding
                                                           Dx Dyv − Dxv Dy   Y
                                               Mh ¼ −                       ¼ hk
                                                           Dxh Dyv − Dxv Dyh X hk
                                                           Dxh Dy − Dx Dyh   Y
                                               Mv ¼ −                       ¼ vk :                                      ðA35Þ
                                                           Dxh Dyv − Dxv Dyh X vk

Once again, the coefficients M can be inferred from the slope of their linear fit Y hk ¼ Mh X hk and Y vk ¼ Mv X vk , with Y
and X evaluated at all BPMs.
   In order for the element Mh;v to be observable one step is actually missing. The signed amplitudes S of Eq. (A29) depend
                                                                          ðqÞ
on: (i) measurable amplitude and phase of BPM modes, fhzjf q ig and ψ z , where z is either x or y and q denotes a generic
mode; (ii) the steerer phases ψ h and ψ v , which are measurable from the BPM data via Eq. (A17); (iii) the quadrupole phase
ψ s , which is yet unknown. The latter can be however inferred from the all other phases by means of Eq. (A28), since
                                       ðhþsÞ                                  ðvþsÞ
                               ψ s ¼ ψ xk       − ψ v − mkj π         ψ s ¼ ψ yk      − ψ h − pkj π
                                            ðh−sÞ                                  ðv−sÞ
                               ψ s ¼ −ψ xk          þ ψ v þ mkj π     ψ s ¼ −ψ yk          þ ψ h þ pkj π
                                       ðvþsÞ                                  ðhþsÞ
                               ψ s ¼ ψ xk      − ψ v − nkj π          ψ s ¼ ψ yk       − ψ h − qkj π                    ðA36Þ
                                            ðv−sÞ                                  ðh−sÞ
                               ψ s ¼ −ψ xk          þ ψ v þ nkj π     ψ s ¼ −ψ yk          þ ψ h þ qkj π
                                       ðsÞ                                    ðsÞ
                               ψ s ¼ ψ xk − okj π;                    ψ s ¼ ψ yk − rkj π;

where the second system is derived from the same analysis in the vertical plane. None of the integer number (mkj , nkj , okj ,
pkj , qkj and rkj ) needs to be evaluated. indeed, by multiplying all above equations by two and taking only the module of 2π,
the following expressions for ψ s are derived
                                       ðhþsÞ                                               ðvþsÞ
                       ψ s ¼ mod ð2ψ xk        − 2ψ h ; 2πÞ=2        ψ s ¼ mod ð2ψ yk              − 2ψ v ; 2πÞ=2
                                      ðh−sÞ                                         ðv−sÞ
                       ψ s ¼ mod ð−2ψ xk þ 2ψ h ; 2πÞ=2              ψ s ¼ mod ð−2ψ yk þ 2ψ v ; 2πÞ=2
                                       ðvþsÞ                                               ðhþsÞ
                       ψ s ¼ mod ð2ψ xk        − 2ψ v ; 2πÞ=2        ψ s ¼ mod ð2ψ yk              − 2ψ h ; 2πÞ=2       ðA37Þ
                                          ðv−sÞ                                              ðh−sÞ
                       ψ s ¼ mod ð−2ψ xk            þ 2ψ v ; 2πÞ=2   ψ s ¼ mod ð−2ψ yk               þ 2ψ h ; 2πÞ=2
                                     ðsÞ                                           ðsÞ
                       ψ s ¼ mod ð2ψ xk ; 2πÞ=2;                     ψ s ¼ mod ð2ψ yk ; 2πÞ=2;
The above expressions can be eventually properly averaged among themselves and over all BPMs to increase the final
accuracy.

                  APPENDIX B: DISCRETE SIGNAL CORRECTED FOURIER COMPONENT
   This appendix details the procedure (already discussed in Ref. [14]) to extract the Fourier component at the frequency f m
of a discrete signal xðtn Þ, sampled at equally spaced time intervals tn ¼ nΔt. This harmonic corresponds to the projection of
x on f m , hxjf m i, introduced in Eq. (A2). This being a complex number, calculations will be made explicit for the real and
                                                                              ðmÞ
imaginary parts separately. These are then used to compute the phase ψ x needed to evaluate the OCM phase ψ h via
Eq. (A16). We assume that the signal xðtn Þ contains a set of N f harmonics at frequencies f m , which are those excited during
the FBBA. The signal can thus be expanded as

                                                               012802-21
MARTÍ, BENEDETTI, IRISO, and FRANCHI                                                 PHYS. REV. ACCEL. BEAMS 23, 012802 (2020)

                                     X
                                     Nf                                         X
                                                                                 Nf                        
                                                               ðmÞ                          ðmÞ
                        xðtn Þ ¼           Xm cosð2πf m tn þ ψ x Þ ¼ ℜ              ðXm eiψ x
                                                                                                Þei2πfm tn

                                     m¼0                                          m¼0
                                    X                     
                                                               1X
                                      Nf                          Nf
                                  ¼ℜ     hxjf m iei2πfm tn
                                                             ¼       ðhxjf m iei2πfm tn þ hxjf m i e−i2πfm tn Þ:                         ðB1Þ
                                     m¼0
                                                               2 m¼0

The index m ¼ 0 is included to account for the dc component too (f m ¼ 0). The first step in computing hxjf m i is to evaluate
the raw Fourier component at a given frequency f r

                                2X N −1
                                                           1XN −1 X
                                                                  Nf
                hxjf r iraw ¼           xðtn Þe−2πifr tn ¼             ðhxjf m iei2πfm tn þ hxjf m i e−i2πfm tn Þe−2πifr tn
                                N n¼0                      N n¼0 m¼0
                                 Nf            X                                   X                          
                                X               1 N −1 −i2πðfr −fm Þtn                     N −1
                                                                                     1          −i2πðf r þfm Þtn
                              ¼       hxjf m i           e               þ hxjf m i             e                   :                     ðB2Þ
                                m¼0
                                                N n¼0                                   N n¼0

The signal sampled at 10 kHz implies that tn ¼ nΔt, with Δt ¼ 0.1 ms. N in the above sums represents the length of FA
vector data containing the x signal. The exponential sums within the above parentheses can be written as

                       X
                       N −1
                                         1 − e−iNΔt         1XN −1
                                                                                        1 1 − e−i2πðfr fm ÞNΔt
                              e−inΔt ¼              ⇒ ξ
                                                          ¼        e−i2πðf r −f m Þtn
                                                                                      ¼                         :                         ðB3Þ
                                          1 − e−iΔt
                                                       rm
                       n¼0
                                                            N n¼0                       N 1 − e−i2πðfr fm ÞΔt

The measured raw projection hxjf r iraw is then a linear combination of all corrected Fourier projections hxjf m i
                                                                                                                                
                                                        ℜfhxjf r ig                 ℜfξþ     −      þ     −
                                                                                       rm þ ξrm gℑfξrm − ξrm g        ℜfhxjf m ig
   hxjf r iraw ¼ ξ−rm hxjf m i þ ξþ          
                                  rm hxjf m i ⇒                             ¼                                                         :   ðB4Þ
                                                        ℑfhxjf r ig   raw           ℑfξþ     −      −     þ
                                                                                       rm þ ξrm gℜfξrm − ξrm g         ℑfhxjf m ig

The above system is then extended to all N f raw projections [measured via Eq. (B2)], resulting in a square linear system
                                               !         !     !         !
                                               hxjfiraw ¼ Chxjfi ⇒ hxjfi ¼ C−1 hxjfiraw                                                   ðB5Þ
             !       !
where both hxjfi and hxjfiraw are vectors of 2N f elements and C is a 2N f × 2N f matrix dependent on all ξ  rm terms. C is
close to the identity matrix, with nonzero off-diagonal elements generated by the finite sampling time Δt and introducing a
cross-talk between all modes. The corrected (i.e., uncoupled) projections hxjfi can be however inferred from the last
(inverted) system of Eq. (B5).




 [1] A. Wolski and F. Zimmermann, Closed orbit response to                  [5] K. R. Schmidt, Misalignments from K-modulation,
     quadrupole strength variation, Lawrence Berkeley National                  Proceedings of The Third Workshop on LEP Perfor-
     Lab. (LBNL) Report No. LBNL-54360 internal note, 2004.                     mance, Chamonix, Franc, edited by J. Poole (1993),
 [2] K. Endo, H. Fukuma, and F. Q. Zhang, Preliminary orbit                     p. 139–145.
     measurement for beam-based alignment, Proceedings of                   [6] G.Portmann, D.Robin, and L.Schachinger, Automated
     EPAC1996, Sitges, Spain, edited by S. Myers, A. Paheco,                    beam based alignment of the ALS quadrupoles, Proceed-
     R. Pascual, C. Petit-Jean-Genaz, and J. Poole (1996),                      ings of EPAC1996, Sitges, Spain, edited by S. Myers, A.
     p. 1657–1659.                                                              Paheco, R. Pascual, C. Petit-Jean-Genaz, and J. Poole
 [3] I. Pinayev, Centering of quadrupole family, Nucl. Instrum.                 (1996), p. 2693–2695.
     Methods Phys. Res., Sect. A 570, 351 (2007).                           [7] A. Madur and P. Brunelle, and A. Nadji, Beam based
 [4] J. Niedziela, C. Montag, and T. Satogata, Quadrupole                       alignment for the storage ring multipoles of synchrotron
     beam-based alignment at RHIC, in Proceedings of the                        soleil, in Proceedings of the 10th European Particle
     21st Particle Accelerator Conference, Knoxville, TN, 2005                  Accelerator Conference, Edinburgh, Scotland, 2006
     (IEEE, Piscataway, NJ, 2005), p. 3493–3495.                                (EPS-AG, Edinburgh, Scotland, 2006), p. 1939–1941.


                                                                  012802-22
FAST BEAM-BASED ALIGNMENT …                                             PHYS. REV. ACCEL. BEAMS 23, 012802 (2020)

 [8] M. D. Woodley, J. Nelson, M. Ross, J. Turner, A. Wolski,          Proceedings of ICALEPCS2011, Grenoble, France, edited
     and K. Kubo, Beam based alignment at the KEK-ATF                  by M. Robichon (2011), pp. 1244–1246.
     damping ring, in Proceedings of the 9th European Particle    [12] A. Olmos and U. Iriso, Feedback systems at ALBA,
     Accelerator Conference, Lucerne, 2004 (EPS-AG,                    Diagnostics Experts European Light Sources DEELS2015,
     Lucerne, 2004) http://accelconf.web.cern.ch/AccelConf/            https://indico.cells.es/indico/event/22/.
     e04/, p. 36–38.                                              [13] S. Gurov et al., ALBA storage ring quadrupoles and
 [9] M. Munoz, Z. Marti, D. Einfeld, and G. Benedetti,                 sextupoles manufacturing and measurements, in Proceed-
     Orbit studies during ALBA commissioning, in Proceed-              ings of the 23rd Particle Accelerator Conference, Van-
     ings of the 2nd International Particle Accelerator                couver, Canada, 2009 (IEEE, Piscataway, NJ, 2009),
     Conference, San Sebastián, Spain (EPS-AG, Spain,                  pp. 160–162.
     2011), p. 3020–3022.                                         [14] Z. Martí, G. Benedetti, M. Carlà, J. Fraxanet, U. Iriso, J.
[10] P. Tenenbaum and T. O. Raubenheimer, Resolution and               Moldes, A. Olmos, and R. Petrocelli, Fast orbit response
     systematic limitations in beam-based alignment, Phys. Rev.        matrix measurements at ALBA, Proceedings of IPAC2017,
     Accel. Beams 3, 052801 (2000).                                    Copenhagen, Denmark, edited by J. Pranke (2017),
[11] M. G. Abbott, G. Rehm, and I. S. Uzun, A new fast                 p. 365–367.
     data logger and viewer at diamond: The FA archiver,




                                                           012802-23
