# Figure guide

Episode panels show the transmitter at the relative origin in top and side projections. Blue and red identify the transmitter and receiver. Solid arrows are velocity vectors scaled by 3 s, dashed arrows are commanded boresights, and the dotted line is the line of sight.

The CSI row labels the three feedback ages. Its color scale is SINR in dB clipped to the P1--P99 range, so limits can differ between images.

The model-transmission panel overlays the latest CSI with the transmitted-bin mask. The selected K is one common repetition factor for the transmitted group; hatched bins are not transmitted. The required-K panel is an offline future reference, while the label-assigned-K panel shows the stored label action. Required K values unreachable within eight repetitions are marked separately, and hatched label bins are not served.

Overview panels show distributions across the demo episodes and the proportion of windows whose stored labels are feasible at each budget. Label feasibility is a property of the stored labels, not model reliability.
