/*
 * @file Ab_initio_Argon_Kokkos inspired by LJFunctorKokkos
 * @date 27.10.26
 * @author Maximilian Zoller
 */

#pragma once

#ifdef AUTOPAS_ENABLE_KOKKOS

#include "autopas/utils/ArrayMath.h"
#include "autopas/baseFunctors/KokkosFunctor.h"
#include "autopas/baseFunctors/PairwiseFunctor.h"

namespace mdLib
{
    template <class Particle_T, bool applyShift = false, bool useMixing = false,
              autopas::FunctorN3Modes useNewton3 = autopas::FunctorN3Modes::Both, bool calculateGlobals = false,
              bool countFLOPs = false, bool relevantForTuning = true>

    class Ab_Initio_Argon_Kokkos : public autopas::PairwiseFunctor<
                                       Particle_T, Ab_Initio_Argon_Kokkos<
                                           Particle_T, applyShift, useMixing, useNewton3, calculateGlobals, countFLOPs,
                                           relevantForTuning>>,
                                   public autopas::KokkosFunctor
    {
    public:

        /**
       * Structure of the SoAs defined by the particle
       */
        using SoAArraysType = typename Particle_T::SoAArraysType;

        /**
         * Precision of SoA entries
         */
        using FloatPrecision = typename Particle_T::ParticleSoAFloatPrecision;

        explicit Ab_Initio_Argon_Kokkos(double cutoff) : autopas::PairwiseFunctor<
                                                             Particle_T, Ab_Initio_Argon_Kokkos>(cutoff),
                                                         KokkosFunctor(),
                                                         _cutoffSquared{static_cast<FloatPrecision>(cutoff * cutoff)}
        {}

        void AoSFunctor(Particle_T &i, Particle_T &j, bool newton3) final
        {
            using namespace autopas::utils::ArrayMath::literals;
            if (i.isDummy() or j.isDummy()) {
                return;
            }
            const FloatPrecision sigmaSquared = 1.;  // TODO: extract that somehow somewhere else
            const FloatPrecision epsilon24 = 24.;    // TODO: extract that somehow somewhere else

            const auto dr = i.getR() - j.getR();
            const double dr2 = autopas::utils::ArrayMath::dot(dr, dr);

            if (dr2 > _cutoffSquared) {
                return;
            }

            const FloatPrecision r = sqrt(dr2);
            const FloatPrecision br = b * r;
            const FloatPrecision e_br = exp(-br);
            const FloatPrecision inv_dr = 1. / r;
            const FloatPrecision inv_dr2 = 1. / dr2;

            FloatPrecision outerSum = 0.;
            FloatPrecision innerSum = 0.;
            for (int n = 3; n <= 8; n++)
            {
                for (int k = 0 ; k <= 2 * n ; k++)
                {
                    innerSum +=  (Kokkos::pow(br, k)/Fak[k]);
                    innerSum +=  (br * (Kokkos::pow(br, 2 * k)/Fak[(2 * n)]));
                }
                outerSum +=  ((C_2n[n-3])/(Kokkos::pow(r, ((2 * n) + 2)))) -e_br * (2 * n * (e_br * innerSum));

                innerSum = 0.;
            }

            //reusing inner sum for final calculations with A*exp()
            const double  Aexpa = A * Kokkos::exp( ((a1 * r) + (a2 * dr2) + (a_1 * inv_dr) + (a_2 * inv_dr2)) );
            innerSum = Aexpa * ((a1 * inv_dr) + (2 * a2) - (a_1 * inv_dr2 * inv_dr) - (2 * a_2 * inv_dr2 * inv_dr2));
            // TODO inv_dr for "normierung" notwendig?
            const double fac =  (innerSum - outerSum) * inv_dr;
            auto f = dr * fac;
            i.addF(f);
            if (newton3)
            {
                //only for newton 3
                j.subF(f);
            }


        }

        constexpr static bool globalCalculationRequested() { return calculateGlobals; }

        double getPotentialEnergy() const { return _potentialEnergy; }

        double getVirial() const { return _virial; }

        void SoAFunctorSingle(autopas::SoAView<SoAArraysType> soa, bool newton3) final {
            // No Op, TODO: make sure this is never used (also not in remainder traversal)

            if (soa.size() == 0) {
                return;
            }

            std::cout << "Trying to call non-existing function LJFunctorKokkos::SoAFunctorSingle" << std::endl;
        }

        void SoAFunctorPair(autopas::SoAView<SoAArraysType> soa1, autopas::SoAView<SoAArraysType> soa2, bool newton3) final {
            // No Op, TODO: make sure this is never used (also not in remainder traversal)

            if (soa1.size() == 0 or soa2.size() == 0) {
                return;
            }

            std::cout << "Trying to call non-existing function LJFunctorKokkos::SoAFunctorPair" << std::endl;
        }

        KOKKOS_INLINE_FUNCTION
        void ForceKernelKokkos(const FloatPrecision &x1, const FloatPrecision &y1, const FloatPrecision &z1,
                               const autopas::utilsKokkos::KokkosStorage<Particle_T> &storage2, FloatPrecision &fxAcc,
                               FloatPrecision &fyAcc, FloatPrecision &fzAcc, FloatPrecision &virialSum,
                               FloatPrecision &uPotSum, FloatPrecision cutoffSquared, int i, int j) final {
            const auto owned2 = storage2.template operator()<Particle_T::AttributeNames::ownershipState, false>(j);

            // TODO: compared to the floating point ifs, this takes much more time to execute (consider having ownershipState as float with guaranteed to be exact values)
            if (owned2 != autopas::OwnershipStateFast::dummy) {
                const auto x2 = storage2.template operator()<Particle_T::AttributeNames::posX, false>(j);
                const auto y2 = storage2.template operator()<Particle_T::AttributeNames::posY, false>(j);
                const auto z2 = storage2.template operator()<Particle_T::AttributeNames::posZ, false>(j);

                const FloatPrecision drX = x1 - x2;
                const FloatPrecision drY = y1 - y2;
                const FloatPrecision drZ = z1 - z2;

                const FloatPrecision dr2 = drX * drX + drY * drY + drZ * drZ;

                if (dr2 <= cutoffSquared && dr2 > 0.) {
                    //TODO: calculate ab initio potentials should be done
                    FloatPrecision fac = 0.;
                    FloatPrecision uPot = 0.;
                    ab_in_arg_pair(dr2, fac, uPot);

                    fxAcc += fac * drX;
                    fyAcc += fac * drY;
                    fzAcc += fac * drZ;

                    if constexpr (calculateGlobals) {
                        uPotSum += uPot;
                    }
                }

            } // if owned2 is dummy
        }

        constexpr static auto getNeededAttr() {
            return std::array<typename Particle_T::AttributeNames, 9>{
                Particle_T::AttributeNames::id,
                Particle_T::AttributeNames::posX,
                Particle_T::AttributeNames::posY,
                Particle_T::AttributeNames::posZ,
                Particle_T::AttributeNames::forceX,
                Particle_T::AttributeNames::forceY,
                Particle_T::AttributeNames::forceZ,
                Particle_T::AttributeNames::typeId,
                Particle_T::AttributeNames::ownershipState,
            };
        }

        constexpr static auto getNeededAttr(std::false_type) {
            return std::array<typename Particle_T::AttributeNames, 6>{
                Particle_T::AttributeNames::id,     Particle_T::AttributeNames::posX,
                Particle_T::AttributeNames::posY,   Particle_T::AttributeNames::posZ,
                Particle_T::AttributeNames::typeId, Particle_T::AttributeNames::ownershipState};
        }

        constexpr static auto getComputedAttr() {
            return std::array<typename Particle_T::AttributeNames, 3>{
                Particle_T::AttributeNames::forceX, Particle_T::AttributeNames::forceY, Particle_T::AttributeNames::forceZ};
        }

        /* Interface required stuff */
        std::string getName() final { return "LJFunctorKokkos"; }

        bool isRelevantForTuning() final { return true; }

        bool allowsNewton3() final {
            // TODO: allow also Newton3 (will require changes in other parts of the program)
            return false;
        }

        bool allowsNonNewton3() final {
            // TODO
            return true;
        }

    private:
        FloatPrecision _cutoffSquared;
        const FloatPrecision A = 46133014.6;
        const FloatPrecision b = 40.2517211;
        const FloatPrecision a1 = - 29.8337630;
        const FloatPrecision a2 = - 9.71208881;
        const FloatPrecision a_1 = 0.0275206827;
        const FloatPrecision a_2 = - 0.0101489050;
        //C_6 - C_16
        const std::array<FloatPrecision, 6> C_2n = {
            0.442812017,
            0.0326707684,
            0.00245656537,
            0.000188246247,
            0.0000147012192,
            0.0000117006343
        };
        const std::array<FloatPrecision, 17> Fak = {
            1., //0!
            1., //1!
            2., //2!
            6., //3!
            24., //4!
            120., //5!
            720., //6!
            5040., //7!
            40320., //8!
            362880., //9!
            3628800., //10!
            39916800., //11!
            479001600., //12!
            6227020800., //13!
            87178291200., //14!
            1307674368000., //15!
            20922789888000., //16!
        };


        KOKKOS_INLINE_FUNCTION
        void ab_in_arg_pair(FloatPrecision dr2, FloatPrecision &fac, FloatPrecision &uPot)
        {
            const FloatPrecision r = Kokkos::sqrt(dr2);
            const FloatPrecision br = b * r;
            const FloatPrecision e_br = Kokkos::exp(-br);
            const FloatPrecision inv_dr = 1. / r;
            const FloatPrecision inv_dr2 = 1. / dr2;

            FloatPrecision outerSum = 0.;

            FloatPrecision innerSum = 0.;

            for (int n = 3; n <= 8; n++)
            {
                for (int k = 0 ; k <= 2 * n ; k++)
                {
                    innerSum +=  (Kokkos::pow(br, k)/Fak[k]);
                    innerSum +=  (br * (Kokkos::pow(br, 2 * k)/Fak[(2 * n)]));
                }
                outerSum +=  ((C_2n[n-3])/(Kokkos::pow(r, ((2 * n) + 2)))) -e_br * (2 * n * (e_br * innerSum));

                innerSum = 0.;
            }

            //reusing inner sum for final calculations with A*exp()
            const FloatPrecision Aexpa = A * Kokkos::exp( ((a1 * r) + (a2 * dr2) + (a_1 * inv_dr) + (a_2 * inv_dr2)) );
            innerSum = Aexpa * ((a1 * inv_dr) + (2 * a2) - (a_1 * inv_dr2 * inv_dr) - (2 * a_2 * inv_dr2 * inv_dr2));
            // TODO inv_dr for "normierung" notwendig?
            fac =  (innerSum - outerSum) * inv_dr;
            if constexpr (calculateGlobals)
            {
                //calculate u(r)
                for (int n = 3; n <= 8; n++)
                {
                    for (int k = 0 ; k <= 2 * n ; k++)
                    {
                        innerSum +=  (Kokkos::pow(br, k)/Fak[k]);
                    }
                    outerSum = ((C_2n[n-3])/(Kokkos::pow(r, (2 * n)))) - (1 - (e_br * innerSum)) ;
                    innerSum = 0.;
                }
                uPot = Aexpa - outerSum;
            }
        }
    };

}

#endif