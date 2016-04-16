# Changes by Release

## 0.4.0 (2016-04-16)

- Nock API is now based on the VM trait with user hooks in calls and
  hints for implementing a jet system.

- Nouns cache mug hashes that match then ones used in current Urbit.

- More conversion implementations between Rust types and nouns.

- Default noun printer abbreviates large nouns.

## 0.3.0 (2016-03-21)

- New Noun structure that supports hidden internal features.

- Introduce conversion traits between Rust types and Nouns.

## 0.2.1 (2016-01-17)

- Relicense to dual MIT or Apache-2.0 as per Rust project licensing
  guidelines.

- Memoizing fold method allows quick computation over nouns which
  would be enormous if iterated naively.

## 0.2.0 (2016-01-10)

- New Noun structure that handles arbitrary-sized atoms and can
  reuse memory for repeating subnouns.

- Nocking API now takes separate subject and formula nouns.

## 0.1.0 (2015-10-25)

- Initial limited toy version that handles basic Nock rules.
