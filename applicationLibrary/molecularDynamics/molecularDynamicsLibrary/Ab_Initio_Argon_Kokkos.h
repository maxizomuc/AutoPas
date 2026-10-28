/*
 * @file Ab_initio_Argon_Kokkos inspired by LJFunctorKokkos
 * @date 27.10.26
 * @author Maximilian Zoller
 */

#pragma once

#ifdef AUTOPAS_ENABLE_KOKKOS

#include "autopas/baseFunctors/KokkosFunctor.h"
#include "autopas/baseFunctors/PairwiseFunctor.h"

namespace mdLib {

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

    explicit Ab_Initio_Argon_Kokkos (double cutoff) : autopas::PairwiseFunctor<Particle_T, Ab_Initio_Argon_Kokkos>(cutoff), KokkosFunctor(), _cutoffSquared{static_cast<FloatPrecision>(cutoff * cutoff)}
    {
    }
    void AoSFunctor(Particle_T &i, Particle_T &j, bool newton3) final
    {
        //using namespace autopas::utils::ArrayMath::literals;
        if (i.isDummy() or j.isDummy()) {
            return;
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
};

}

#endif