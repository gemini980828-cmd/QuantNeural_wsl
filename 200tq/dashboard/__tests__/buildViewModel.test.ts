import { describe, expect, it } from "vitest";

import { buildViewModel, E03RawInputs } from "@/lib/ops/e03/buildViewModel";

function makeInputs(overrides: Partial<E03RawInputs> = {}): E03RawInputs {
  return {
    now: new Date("2026-02-11T00:00:00.000Z"),
    dataState: "FRESH_CLOSE",
    emergencyState: "NONE",
    executionState: "SCHEDULED",
    sma3: 480,
    sma160: 450,
    sma165: 455,
    sma170: 460,
    lastUpdatedKst: "2026-02-11 06:00:00",
    verdictDateKst: "2026-02-10",
    executionDateKst: "2026-02-11",
    simulationMode: false,
    privacyMode: false,
    inputHoldings: { TQQQ: 500, SGOV: 100 },
    inputPrices: { TQQQ: 100, SGOV: 100 },
    ...overrides,
  };
}

describe("buildViewModel v2026.3", () => {
  it("ON-Normal with flipCount=0 stays ON with 100% weight", () => {
    const vm = buildViewModel(makeInputs({ signalHistory: [true, true, true, true] }));

    expect(vm.strategyState).toBe("ON");
    expect(vm.flipCount).toBe(0);
    expect(vm.targetTqqqWeight).toBe(1);
    expect(vm.isChoppy).toBe(false);
  });

  it("ON-Normal with flipCount=2 stays ON", () => {
    const vm = buildViewModel(makeInputs({ signalHistory: [true, false, false, true, true] }));
    expect(vm.flipCount).toBe(2);
    expect(vm.strategyState).toBe("ON");
  });

  it("ON-Choppy with flipCount=3 becomes ON_CHOPPY", () => {
    const vm = buildViewModel(makeInputs({ signalHistory: [true, false, true, false, false] }));

    expect(vm.flipCount).toBe(3);
    expect(vm.strategyState).toBe("ON_CHOPPY");
    expect(vm.targetTqqqWeight).toBe(0.7);
  });

  it("ON-Choppy with flipCount=5 remains ON_CHOPPY", () => {
    const vm = buildViewModel(makeInputs({ signalHistory: [true, false, true, false, true, false] }));

    expect(vm.flipCount).toBe(5);
    expect(vm.strategyState).toBe("ON_CHOPPY");
  });

  it("OFF state when vote is OFF regardless of low flip count", () => {
    const vm = buildViewModel(
      makeInputs({
        sma3: 400,
        signalHistory: [false, false, false, false],
      }),
    );

    expect(vm.strategyState).toBe("OFF10");
    expect(vm.targetTqqqWeight).toBe(0.1);
  });

  it("OFF state ignores choppy filter even with high flip count", () => {
    const vm = buildViewModel(
      makeInputs({
        sma3: 400,
        signalHistory: [true, false, true, false, true, false],
      }),
    );

    expect(vm.flipCount).toBe(5);
    expect(vm.strategyState).toBe("OFF10");
    expect(vm.isChoppy).toBe(false);
  });

  it("Emergency QQQ trigger has priority over ON", () => {
    const vm = buildViewModel(makeInputs({ qqqDailyReturn: -6 }));
    expect(vm.strategyState).toBe("EMERGENCY");
  });

  it("Emergency TQQQ trigger has priority over OFF", () => {
    const vm = buildViewModel(
      makeInputs({
        sma3: 400,
        tqqqEntryPrice: 120,
        inputPrices: { TQQQ: 100, SGOV: 100 },
      }),
    );

    expect(vm.strategyState).toBe("EMERGENCY");
  });

  it("Cooldown active forces OFF10 even when ON signal", () => {
    const vm = buildViewModel(makeInputs({ cooldownActive: true }));
    expect(vm.strategyState).toBe("OFF10");
    expect(vm.cooldownActive).toBe(true);
  });

  it("calculates flipCount from signalHistory transitions", () => {
    const vm = buildViewModel(
      makeInputs({
        signalHistory: [true, true, false, true, false, true],
      }),
    );

    expect(vm.flipCount).toBe(4);
  });

  it("missing signalHistory defaults flipCount to 0 without choppy", () => {
    const vm = buildViewModel(makeInputs({ signalHistory: undefined }));

    expect(vm.flipCount).toBe(0);
    expect(vm.isChoppy).toBe(false);
    expect(vm.strategyState).toBe("ON");
  });

  it("ON_CHOPPY generates 70/30 rebalance trade lines", () => {
    const vm = buildViewModel(
      makeInputs({
        signalHistory: [true, false, true, false, false],
        inputHoldings: { TQQQ: 500, SGOV: 100 },
        inputPrices: { TQQQ: 100, SGOV: 100 },
      }),
    );

    expect(vm.strategyState).toBe("ON_CHOPPY");
    expect(vm.expectedTrades.length).toBeGreaterThanOrEqual(2);
    expect(vm.expectedTrades[0].ticker).toBe("TQQQ");
    expect(vm.expectedTrades[0].note).toContain("70%");
  });

  it("EMERGENCY generates OFF10-equivalent trades with emergency note", () => {
    const vm = buildViewModel(
      makeInputs({
        qqqDailyReturn: -6,
        inputHoldings: { TQQQ: 500, SGOV: 100 },
      }),
    );

    const sellTqqq = vm.expectedTrades.find((t) => t.ticker === "TQQQ" && t.action === "SELL");
    expect(vm.strategyState).toBe("EMERGENCY");
    expect(sellTqqq?.note).toContain("Emergency Exit");
  });

  it("sets verdict titles for all 4 states", () => {
    const onVm = buildViewModel(makeInputs());
    const choppyVm = buildViewModel(makeInputs({ signalHistory: [true, false, true, false, false] }));
    const offVm = buildViewModel(makeInputs({ sma3: 400 }));
    const emergencyVm = buildViewModel(makeInputs({ qqqDailyReturn: -6 }));

    expect(onVm.verdictTitle).toBe("VERDICT: ON");
    expect(choppyVm.verdictTitle).toBe("VERDICT: ON (CHOPPY)");
    expect(offVm.verdictTitle).toBe("VERDICT: OFF10");
    expect(emergencyVm.verdictTitle).toBe("VERDICT: EMERGENCY");
  });

  it("keeps backward-compatible ON/OFF behavior when new fields are missing", () => {
    const onVm = buildViewModel(makeInputs({ signalHistory: undefined }));
    const offVm = buildViewModel(makeInputs({ sma3: 400, signalHistory: undefined }));

    expect(onVm.strategyState).toBe("ON");
    expect(offVm.strategyState).toBe("OFF10");
  });
});
