import React, { useEffect, useRef } from "react";
import { animated } from "@react-spring/web";
import styled from "styled-components";
import { summarizeLongStr } from "../../utils/generic";

import { useHighlightHover } from "../Actions/utils";
import { ItemAction } from "../Actions/ItemAction";
import { getValueString } from "../Filters/utils";

export const ResultsContainer = styled(animated.div)`
  background-color: ${({ theme }) => theme.backgroundDark};
  border: 1px solid ${({ theme }) => theme.backgroundDarkBorder};
  border-radius: 2px;
  box-shadow: 0 2px 20px ${({ theme }) => theme.backgroundDark};
  box-sizing: border-box;
  margin-top: 0;
  position: absolute;
  width: auto;
  z-index: 801;
  padding: 0 0.5rem;
  width: calc(100% - 12px);
  left: 6px;
  margin-bottom: 1rem;
`;

const ResultDiv = styled(ItemAction)`
  white-space: nowrap;
  overflow: hidden;
  display: flex;
  text-overflow: ellipsis;
  margin: 0;
  justify-content: space-between;
  flex-direction: row;
`;

const ScrollResultsContainer = styled.div`
  margin-left: -0.5rem;
  margin-right: -0.5rem;
  max-height: 330px;
  overflow-y: scroll;
  scrollbar-width: none;
  &::-webkit-scrollbar {
    width: 0px;
    background: transparent;
    display: none;
  }
  &::-webkit-scrollbar-thumb {
    width: 0px;
    display: none;
  }
`;

type ResultValue<T> = [T, number];

interface ResultProps<T> {
  result: T;
  count: number;
  highlight: string;
  active: boolean | null;
  onClick: () => void;
  maxLen?: number;
  color: string;
}

const Result = React.memo(
  <T extends unknown>({
    active,
    highlight,
    onClick,
    maxLen,
    color,
    result,
    count,
  }: ResultProps<T>) => {
    const props = useHighlightHover(
      false,
      active ? active : null,
      result === null ? highlight : null
    );
    const ref = useRef<HTMLDivElement>();
    const wasActive = useRef(false);

    const [text, coloring] = getValueString(result);

    useEffect(() => {
      if (active && ref.current && !wasActive.current) {
        ref.current.scrollIntoView(true);
        wasActive.current = true;
      } else if (!active) {
        wasActive.current = false;
      }
    }, [active]);

    return (
      <ResultDiv
        title={result === null ? "None" : result}
        {...props}
        onClick={onClick}
        ref={ref}
      >
        <span style={coloring ? { color } : {}}>
          {maxLen ? summarizeLongStr(text, maxLen, "middle") : text}
        </span>
        {typeof count === "number" && <span>{count.toLocaleString()}</span>}
      </ResultDiv>
    );
  }
);

interface ResultsProps<T> {
  results: ResultValue<T>[];
  highlight: string;
  onSelect: (value: T) => void;
  active: string | null;
  alignRight?: boolean;
  color: string;
}

const Results = React.memo(
  <T extends unknown>({
    color,
    onSelect,
    results,
    highlight,
    active = undefined,
  }: ResultsProps<T>) => {
    return (
      <ScrollResultsContainer>
        {results.map((result) => (
          <Result
            key={String(result[0])}
            result={result[0]}
            count={result[1]}
            highlight={highlight}
            onClick={() => onSelect(result[0])}
            active={active === result[0]}
            maxLen={26 - result[1].toLocaleString().length}
            color={color}
          />
        ))}
      </ScrollResultsContainer>
    );
  }
);

export default Results;
