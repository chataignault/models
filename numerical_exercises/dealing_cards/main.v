Require Import Coq.Arith.PeanoNat.
(* Require Import Stdlib.Arith.PeanoNat. *)

Inductive deck : nat -> Set :=
  | nild : deck 0
  | consd : forall n : nat, bool -> deck n -> deck (S n).

Fixpoint flip {n : nat} (d : deck n) : deck n :=
  match d with
    | nild => nild
    | consd i b d' => consd i (negb b) (flip d')
  end.

Fixpoint count_ups {n : nat} (d : deck n) : nat :=
  match d with 
    | nild => 0
    | consd i true d' => S (count_ups d')
    | consd i false d' => (count_ups d')
  end. 

Fixpoint split {n : nat} (d : deck n) (m : nat) (H : m <= n) :
  deck m * deck (n - m) :=
  match d in deck n' return forall m', m' <= n' -> deck m' * deck (n' - m') with
  | nild => fun m' H' => 
      match m' as m'' return m'' <= 0 -> deck m'' * deck (0 - m'') with
      | 0 => fun _ => (nild, nild)
      | S m'' => fun H'' => match Nat.nle_succ_0 m'' H'' with end
      end H'
  | consd n' b d' => fun m' H' =>
      match m' as m'' return m'' <= S n' -> deck m'' * deck (S n' - m'') with
      | 0 => fun _ => 
          (nild, consd n' b d')
      | S m'' => fun H'' =>
          let H''' : m'' <= n' := le_S_n m'' n' H'' in
          let (d1, d2) := split d' m'' H''' in
          (consd m'' b d1, d2)
      end H'
  end m H.

Theorem equal_ups : 
  forall (n : nat) (m : nat) (H: m <= n) (d: deck n), 
  count_ups d = m -> 
  let (d1, d2) := split d m H in 
  count_ups (flip d1) = count_ups d2.
